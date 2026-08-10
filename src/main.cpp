#include "main.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <set>
#include <future>

#include "YoloDetector.h"
#include "rtspServer.h"
#include "ConfigManager.h"
#include "APIController.h"
#include "SourceHandler.h"
#include "CrowdCounting.h"

// ---------------- Logging, prefix [Main] ----------------
void logInfo(const std::string& msg) {
    std::cout << "[Main] [INFO] " << msg << std::endl;
}

void logWarn(const std::string& msg) {
    std::cout << "[Main] [WARN] " << msg << std::endl;
}

void logError(const std::string& msg) {
    std::cerr << "[Main] [ERROR] " << msg << std::endl;
}

std::vector<Detection> applyRuntimeFilter(const std::vector<Detection>& raw,
                                           RuntimeAIConfig& runtimeCfg) {
    float threshold;
    std::set<int> classes;
    {
        std::lock_guard<std::mutex> lock(runtimeCfg.mtx);
        threshold = runtimeCfg.confidenceThreshold;
        classes = runtimeCfg.targetClasses;
    }

    std::vector<Detection> filtered;
    filtered.reserve(raw.size());
    for (const auto& d : raw) {
        if (d.confidence < threshold) continue;
        if (!classes.empty() && classes.count(d.classId) == 0) continue;
        filtered.push_back(d);
    }
    return filtered;
}


int main(int argc, char* argv[]) {
    std::string configPath;
    Config _appConfig = ConfigManager::load(argc, argv, configPath);
    SourceHandler sourceHandler;
    cv::VideoCapture cap;
    {
        int attempt = 0;
        while (true) {
            ++attempt;
            if (sourceHandler.openCapture(_appConfig, cap)) {
                logInfo("Input berhasil dibuka pada percobaan ke-" + std::to_string(attempt));
                break;
            }

            if (attempt == 1 || attempt % 10 == 0) {
                logWarn("Input '" + _appConfig.inputSource + "' belum tersedia (percobaan ke-" +
                        std::to_string(attempt) + "). Mencoba lagi tiap " +
                        std::to_string(_appConfig.reconnectIntervalMs) + " ms...");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(_appConfig.reconnectIntervalMs));
        }
    }

    if (!_appConfig.isGstreamer) {
        int fourccInt = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char fourccStr[5] = {
            static_cast<char>(fourccInt & 0xFF),
            static_cast<char>((fourccInt >> 8) & 0xFF),
            static_cast<char>((fourccInt >> 16) & 0xFF),
            static_cast<char>((fourccInt >> 24) & 0xFF),
            '\0'
        };
        logInfo(std::string("FOURCC aktual: ") + fourccStr +
                " | FPS diminta ke driver: " + std::to_string(cap.get(cv::CAP_PROP_FPS)));
    } else {
        logInfo("Input via GStreamer pipeline, resolusi & fps mengikuti pipeline/source.");
    }

    // ---- Load model YOLO ----
    std::unique_ptr<YoloDetector> detector;
    try {
        detector = std::make_unique<YoloDetector>(
            _appConfig.modelPath, _appConfig.inferSize, _appConfig.confThresh, _appConfig.nmsThresh, _appConfig.targetClasses);
    } catch (const std::exception& e) {
        logError("Gagal load model: " + _appConfig.modelPath + " | Pesan: " + e.what());
        return 1;
    }

    // ---- Modul Crowd Counting, hanya dimuat kalau flag aktif ----
    // Ukuran input & interval bisa diatur dari config.ini supaya gampang
    // di-tuning tanpa perlu rebuild/redeploy binary (lihat catatan FPS
    // drop -- VGG-19 backbone DM-Count jauh lebih berat dari YOLO, jadi
    // resolusi & interval infer perlu di-tuning sesuai kebutuhan).
    std::unique_ptr<CrowdCounting> crowdCounter;
    CrowdCountResult lastCrowdResult;   // cache hasil infer terakhir (dipakai antar-frame saat interval > 1)
    long crowdFrameCounter = 0;

    // Interval berapa frame sekali crowd counting benar-benar dijalankan.
    // 1 = tiap frame (paling berat), 5 = tiap 5 frame (lebih ringan, count
    // ter-update tiap ~5 frame -- cukup untuk monitoring kerumunan yang
    // tidak berubah drastis dalam hitungan ratus milidetik).
    const int crowdInferInterval = (_appConfig.crowdInferInterval > 0)
                                        ? _appConfig.crowdInferInterval
                                        : 5;
    try {
        cv::Size crowdInputSize(
            _appConfig.crowdInputWidth  > 0 ? _appConfig.crowdInputWidth  : 1024,
            _appConfig.crowdInputHeight > 0 ? _appConfig.crowdInputHeight : 768
        );
        crowdCounter = std::make_unique<CrowdCounting>(_appConfig.crowdModelPath, crowdInputSize);
        logInfo("Modul Crowd Counting aktif. Engine: " + _appConfig.crowdModelPath +
                " | input: " + std::to_string(crowdInputSize.width) + "x" + std::to_string(crowdInputSize.height) +
                " | interval: tiap " + std::to_string(crowdInferInterval) + " frame");
    } catch (const std::exception& e) {
        logError("Gagal load model Crowd Counting: " + std::string(e.what()));
        logWarn("Modul Crowd Counting dinonaktifkan, aplikasi tetap berjalan tanpa fitur ini.");
        crowdCounter.reset(); // pastikan null, jangan biarkan pointer setengah-inisialisasi
    }

    if (!_appConfig.isDetection) {
        logInfo("Modul Detection nonaktif (isDetection=false).");
    }
    if (!_appConfig.isCrowdCounting) {
        logInfo("Modul Crowd Counting nonaktif (isCrowdCounting=false).");
    }

    RuntimeAIConfig runtimeAiConfig;
    runtimeAiConfig.confidenceThreshold = _appConfig.confThresh;
    runtimeAiConfig.targetClasses = std::set<int>(_appConfig.targetClasses.begin(), _appConfig.targetClasses.end());

    APIController apiController;
    apiController.setConfigStore(&_appConfig, configPath);

    apiController.setConfigCommandHandler([&](const AIConfig& newCfg) -> bool {
        // ---- Golongan USER: berlaku langsung tanpa restart ----
        {
            std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);

            if (newCfg.confidenceThreshold.has_value()) {
                runtimeAiConfig.confidenceThreshold = static_cast<float>(newCfg.confidenceThreshold.value());
            }
            if (newCfg.classesEnabled.has_value()) {
                runtimeAiConfig.targetClasses.clear();
                for (const auto& s : newCfg.classesEnabled.value()) {
                    try {
                        runtimeAiConfig.targetClasses.insert(std::stoi(s));
                    } catch (const std::exception&) {
                        logWarn("class berisi nilai non-numerik, diabaikan: " + s);
                    }
                }
            }
        }

        // isDetection/isCrowdCounting/showOverlay dibaca langsung dari
        // _appConfig di tiap iterasi loop utama, jadi cukup di-update di sini
        // supaya berlaku live tanpa restart.
        if (newCfg.detectionEnabled.has_value()) {
            _appConfig.isDetection = newCfg.detectionEnabled.value();
            logInfo(std::string("Detection ") + (_appConfig.isDetection ? "diaktifkan" : "dinonaktifkan") + " via cmd.");
        }
        if (newCfg.crowdCountingEnabled.has_value()) {
            _appConfig.isCrowdCounting = newCfg.crowdCountingEnabled.value();
            logInfo(std::string("Crowd Counting ") + (_appConfig.isCrowdCounting ? "diaktifkan" : "dinonaktifkan") + " via cmd.");
        }
        if (newCfg.showOverlay.has_value()) {
            _appConfig.showOverlay = newCfg.showOverlay.value();
        }

        // reconnect_ms juga live -- dibaca langsung dari _appConfig di
        // reconnect loop (captureThread & probe frame), tidak butuh restart.
        if (newCfg.reconnectIntervalMs.has_value()) {
            _appConfig.reconnectIntervalMs = newCfg.reconnectIntervalMs.value();
            logInfo("reconnect_ms diperbarui via cmd: " + std::to_string(_appConfig.reconnectIntervalMs));
        }

        // ---- Golongan SEMI-DEV (crowd_interval, crowd_width, crowd_height,
        // infer_size, bitrate_kbps): TIDAK diterapkan ke objek yang sedang
        // berjalan di sini -- crowdInferInterval, ukuran buffer CrowdCounting,
        // engine YOLO, dan encoder RtspServer semuanya sudah dikonstruksi
        // dengan nilai lama saat startup. persistConfig() di APIController
        // tetap menulis nilai baru ini ke Config master & config.ini, dan
        // baru benar-benar berlaku setelah aplikasi di-restart -- makanya
        // config_ack untuk parameter ini berisi requires_restart=true.

        // ---- Broadcast "config": HANYA dikirim di sini, yaitu saat ada
        // command dari client (bukan berkala di loop utama). Ambil nilai
        // conf/class terbaru dari runtimeAiConfig (sudah di-update di atas).
        // Untuk nms, pakai nilai baru dari command ini kalau dikirim,
        // kalau tidak fallback ke _appConfig.nmsThresh -- karena
        // persistConfig() di APIController baru menulis ke _appConfig
        // SETELAH callback ini selesai (applied == true), jadi belum
        // tentu ter-update di titik ini.
        double statusConfidenceThreshold;
        std::set<std::string> statusClasses;
        {
            std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);
            statusConfidenceThreshold = runtimeAiConfig.confidenceThreshold;
            for (int c : runtimeAiConfig.targetClasses) {
                statusClasses.insert(std::to_string(c));
            }
        }
        double statusNms = newCfg.iouThreshold.has_value()
                                ? newCfg.iouThreshold.value()
                                : _appConfig.nmsThresh;

        apiController.broadcastConfig(statusConfidenceThreshold, statusNms,
                                       statusClasses, _appConfig.showOverlay,
                                       _appConfig.isDetection, _appConfig.isCrowdCounting);

        logInfo("Config AI diperbarui via cmd.");
        return true;
    });
    std::thread apiThread([&apiController, &_appConfig]() {
        apiController.run(static_cast<uint16_t>(_appConfig.apiPort));
    });
    apiThread.detach();

    cv::Mat probeFrame;
    {
        int attempt = 0;
        while (true) {
            ++attempt;
            if (cap.read(probeFrame) && !probeFrame.empty()) {
                break;
            }
            if (attempt == 1 || attempt % 30 == 0) {
                logWarn("Belum berhasil membaca frame dari source (percobaan ke-" +
                        std::to_string(attempt) + "). Menunggu...");
            }
            if (!cap.isOpened()) {
                logWarn("Koneksi ke input terputus saat probe, mencoba membuka ulang...");
                cap.release();
                int reopenAttempt = 0;
                while (!sourceHandler.openCapture(_appConfig, cap)) {
                    ++reopenAttempt;
                    if (reopenAttempt == 1 || reopenAttempt % 10 == 0) {
                        logWarn("Percobaan buka ulang ke-" + std::to_string(reopenAttempt) +
                                " gagal, mencoba lagi tiap " + std::to_string(_appConfig.reconnectIntervalMs) + " ms...");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(_appConfig.reconnectIntervalMs));
                }
                logInfo("Input berhasil dibuka ulang, melanjutkan probe frame.");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
    if (probeFrame.channels() == 4) {
        cv::cvtColor(probeFrame, probeFrame, cv::COLOR_BGRA2BGR);
    }

    const int actualWidth  = probeFrame.cols;
    const int actualHeight = probeFrame.rows;

    if (actualWidth != _appConfig.width || actualHeight != _appConfig.height) {
        logInfo("Resolusi asli dari source: " + std::to_string(actualWidth) + "x" + std::to_string(actualHeight) +
                " (berbeda dari config " + std::to_string(_appConfig.width) + "x" + std::to_string(_appConfig.height) +
                "). Memakai resolusi asli source secara dinamis.");
    } else {
        logInfo("Resolusi asli dari source: " + std::to_string(actualWidth) + "x" + std::to_string(actualHeight));
    }
    _appConfig.width  = actualWidth;
    _appConfig.height = actualHeight;

    RtspServer rtspServer(
        _appConfig.rtspPort,
        _appConfig.rtspMount,
        _appConfig.width,
        _appConfig.height,
        _appConfig.fps,
        _appConfig.ipAdress,
        _appConfig.encoderType,
        _appConfig.codecType,
        _appConfig.bitrateKbps
    );

    if (!rtspServer.start()) {
        logError("Gagal menjalankan RTSP Server.");
        return 1;
    }

    std::mutex captureMutex;
    cv::Mat latestCapturedFrame = probeFrame;
    std::atomic<bool> captureRunning{true};
    std::atomic<bool> frameAvailable{true};
    std::mutex fpsMutex;
    double captureFps = 0.0;

    std::thread captureThread([&]() {
        cv::Mat tmp;
        int frameCount = 0;
        auto fpsWindowStart = std::chrono::steady_clock::now();
        bool resolutionMismatchWarned = false;

        while (captureRunning) {
            if (!cap.read(tmp) || tmp.empty()) {
                logWarn("Gagal ambil frame dari source. Mencoba menyambung ulang...");
                {
                    std::lock_guard<std::mutex> lock(captureMutex);
                    frameAvailable = false;
                }
                cap.release();
                int attempt = 0;
                while (captureRunning && !sourceHandler.openCapture(_appConfig, cap)) {
                    ++attempt;
                    if (attempt == 1 || attempt % 10 == 0) {
                        logWarn("Reconnect percobaan ke-" + std::to_string(attempt) +
                                " gagal, mencoba lagi tiap " + std::to_string(_appConfig.reconnectIntervalMs) + " ms...");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(_appConfig.reconnectIntervalMs));
                }
                if (!captureRunning) break;
                logInfo("Berhasil menyambung ulang ke source, melanjutkan streaming.");
                resolutionMismatchWarned = false;
                continue;
            }

            if (tmp.channels() == 4) {
                cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
            }

            if (tmp.cols != _appConfig.width || tmp.rows != _appConfig.height) {
                if (!resolutionMismatchWarned) {
                    logWarn("Resolusi frame berubah jadi " + std::to_string(tmp.cols) + "x" + std::to_string(tmp.rows) +
                            " (acuan " + std::to_string(_appConfig.width) + "x" + std::to_string(_appConfig.height) +
                            "). Melakukan resize agar stream RTSP tetap konsisten.");
                    resolutionMismatchWarned = true;
                }
                cv::resize(tmp, tmp, cv::Size(_appConfig.width, _appConfig.height));
            }

            {
                std::lock_guard<std::mutex> lock(captureMutex);
                latestCapturedFrame = tmp;
                frameAvailable = true;
            }

            ++frameCount;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - fpsWindowStart).count();
            if (elapsed >= 1.0) {
                std::lock_guard<std::mutex> fpsLock(fpsMutex);
                captureFps = frameCount / elapsed;
                frameCount = 0;
                fpsWindowStart = now;
            }
        }
    });

    cv::Mat frame;
    int displayFrameCount = 0;
    double displayFps = 0.0;
    auto displayFpsWindowStart = std::chrono::steady_clock::now();
    long frameId = 0;

    while (captureRunning) {
        if (!frameAvailable) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(captureMutex);
            if (!frameAvailable) continue;
            frame = latestCapturedFrame.clone();
        }

        // ---- Fork: jalankan Detection dan Crowd Counting secara independen ----
        // Sesuai flag isDetection / isCrowdCounting. Kalau keduanya aktif,
        // dieksekusi paralel via std::async supaya crowd counting yang berat
        // tidak nge-block pipeline deteksi (dan sebaliknya). Masing-masing
        // worker menerima clone frame sendiri karena cv::Mat bukan thread-safe
        // untuk ditulis dari dua sisi sekaligus.
        std::vector<Detection> detections;
        double inferMs = 0.0;
        CrowdCountResult crowdResult = lastCrowdResult; // default: pakai cache frame sebelumnya

        std::future<std::vector<Detection>> detectionFuture;
        std::future<CrowdCountResult> crowdFuture;

        // Crowd counting cuma benar-benar dijalankan tiap crowdInferInterval
        // frame -- VGG-19 backbone DM-Count jauh lebih berat dari YOLO, dan
        // kerumunan tidak berubah drastis dalam hitungan ratus milidetik, jadi
        // tidak perlu di-infer setiap frame. Di frame yang di-skip, overlay tetap
        // pakai lastCrowdResult (cache).
        const bool shouldRunCrowdThisFrame =
            _appConfig.isCrowdCounting && crowdCounter &&
            (crowdFrameCounter % crowdInferInterval == 0);

        auto t0 = std::chrono::steady_clock::now();

        if (_appConfig.isDetection) {
            cv::Mat detFrame = frame; // infer() hanya baca, aman dishare
            detectionFuture = std::async(std::launch::async, [&detector, detFrame]() {
                return detector->infer(detFrame);
            });
        }

        if (shouldRunCrowdThisFrame) {
            cv::Mat crowdFrame = frame.clone(); // clone supaya aman dari race
            crowdFuture = std::async(std::launch::async, [&crowdCounter, crowdFrame]() {
                return crowdCounter->infer(crowdFrame);
            });
        }

        if (_appConfig.isCrowdCounting && crowdCounter) {
            ++crowdFrameCounter; // hitung tiap frame video, bukan tiap infer
        }

        // ---- Join: tunggu kedua worker selesai sebelum lanjut ke postprocessing ----
        std::vector<Detection> rawDetections;
        if (detectionFuture.valid()) {
            rawDetections = detectionFuture.get();
        }
        if (crowdFuture.valid()) {
            crowdResult = crowdFuture.get();
            lastCrowdResult = crowdResult; // update cache untuk frame-frame berikutnya
        }

        auto t1 = std::chrono::steady_clock::now();
        inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::vector<ApiDetectionResult> apiDets;
        ApiCrowdResult apiCrowd;
        // ---- Bbox deteksi: SELALU digambar kalau isDetection aktif,
        // terlepas dari showOverlay. showOverlay hanya mengontrol teks
        // statistik tambahan (count/FPS/resolusi), bukan bbox itu sendiri. ----
        if (_appConfig.isDetection) {
            detections = applyRuntimeFilter(rawDetections, runtimeAiConfig);

            apiDets.reserve(detections.size());
            int idx = 0;
            for (const auto& det : detections) {
                ApiDetectionResult ad;
                ad.id = idx++;
                ad.cls = detector->getClassName(det.classId);
                ad.confidence = det.confidence;
                ad.bbox_x = static_cast<double>(det.box.x) / frame.cols;
                ad.bbox_y = static_cast<double>(det.box.y) / frame.rows;
                ad.bbox_w = static_cast<double>(det.box.width)  / frame.cols;
                ad.bbox_h = static_cast<double>(det.box.height) / frame.rows;
                apiDets.push_back(ad);
            }

            for (const auto& det : detections) {
                cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label =
                    std::string(detector->getClassName(det.classId)) +
                    " " +
                    cv::format("%.2f", det.confidence);
                cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }
        ++frameId;

        // ---- Merge hasil Crowd Counting: heatmap density SELALU digambar
        // kalau modul aktif & hasil valid, terlepas dari showOverlay. Cuma
        // TEKS estimasi jumlah ("Estimasi Kerumunan: N") yang mengikuti
        // flag showOverlay -- sama seperti teks statistik lain. ----
        if (_appConfig.isCrowdCounting && crowdCounter && crowdResult.valid) {
            cv::addWeighted(frame, 1.0, crowdResult.heatmapOverlay, 0.4, 0.0, frame);

            if (_appConfig.showOverlay) {
                std::string crowdText = "Estimasi Kerumunan: " + std::to_string(crowdResult.estimatedCount);
                cv::putText(frame, crowdText, cv::Point(20, 160),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 255), 3);
            }
        }
        apiCrowd.count = crowdResult.estimatedCount;
        apiCrowd.densityLevel = crowdResult.densityLevel;
        apiController.broadcastResult(frameId, frame.cols, frame.rows, apiDets, inferMs, displayFps, apiCrowd);

        ++displayFrameCount;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - displayFpsWindowStart).count();

        if (elapsed >= 1.0) {
            displayFps = displayFrameCount / elapsed;
            displayFrameCount = 0;
            displayFpsWindowStart = now;
            // broadcastConfig TIDAK dikirim di sini lagi -- sekarang cuma
            // dikirim saat ada command "cmd" masuk dari client, lihat
            // setConfigCommandHandler di atas.
        }

        // ---- Teks statistik (count/infer time/FPS/resolusi): HANYA
        // muncul kalau showOverlay aktif. Ini satu-satunya bagian yang
        // dikontrol showOverlay -- bbox & heatmap density tetap tampil
        // di luar blok ini. ----
        if (_appConfig.showOverlay) {
            double captureFpsSnapshot;
            {
                std::lock_guard<std::mutex> fpsLock(fpsMutex);
                captureFpsSnapshot = captureFps;
            }
            std::string countText = "Jumlah Deteksi: " + std::to_string(detections.size()) +
                                     "  |  Infer: " + std::to_string(static_cast<int>(inferMs)) + " ms";
            std::string fpsText = "FPS Sumber: " + std::to_string(static_cast<int>(captureFpsSnapshot)) +
                                   "  |  FPS Output: " + std::to_string(static_cast<int>(displayFps));
            std::string resText = "Resolusi: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows);

            cv::putText(frame, countText, cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 3);
            cv::putText(frame, fpsText, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 3);
            cv::putText(frame, resText, cv::Point(20, 120),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 3);
        }

        rtspServer.pushFrame(frame);
    }

    captureRunning = false;
    if (captureThread.joinable()) captureThread.join();

    cap.release();

    logInfo("Program selesai.");
    return 0;
}