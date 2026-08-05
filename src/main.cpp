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

    if (_appConfig.isCrowdCounting) {
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
    } else {
        logInfo("Modul Crowd Counting nonaktif (isCrowdCounting=false).");
    }

    if (!_appConfig.isDetection) {
        logInfo("Modul Detection nonaktif (isDetection=false).");
    }

    RuntimeAIConfig runtimeAiConfig;
    runtimeAiConfig.confidenceThreshold = _appConfig.confThresh;
    runtimeAiConfig.targetClasses = std::set<int>(_appConfig.targetClasses.begin(), _appConfig.targetClasses.end());

    APIController apiController;
    apiController.setConfigStore(&_appConfig, configPath);

    apiController.setConfigCommandHandler([&](const AIConfig& new_appConfig) -> bool {
        std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);
        runtimeAiConfig.confidenceThreshold = static_cast<float>(new_appConfig.confidence_threshold);
        runtimeAiConfig.targetClasses.clear();
        for (const auto& s : new_appConfig.classes_enabled) {
            try {
                runtimeAiConfig.targetClasses.insert(std::stoi(s));
            } catch (const std::exception&) {
                logWarn("classes_enabled berisi nilai non-numerik, diabaikan: " + s);
            }
        }
        logInfo("Config AI diperbarui via APIController: threshold=" +
                std::to_string(runtimeAiConfig.confidenceThreshold) +
                ", jumlah class aktif=" + std::to_string(runtimeAiConfig.targetClasses.size()));
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
        CrowdCountResult crowdResult = lastCrowdResult; // [BARU] default: pakai cache frame sebelumnya

        std::future<std::vector<Detection>> detectionFuture;
        std::future<CrowdCountResult> crowdFuture;

        // [BARU] Crowd counting cuma benar-benar dijalankan tiap crowdInferInterval
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
            ++crowdFrameCounter; // [BARU] hitung tiap frame video, bukan tiap infer
        }

        // ---- Join: tunggu kedua worker selesai sebelum lanjut ke postprocessing ----
        std::vector<Detection> rawDetections;
        if (detectionFuture.valid()) {
            rawDetections = detectionFuture.get();
        }
        if (crowdFuture.valid()) {
            crowdResult = crowdFuture.get();
            lastCrowdResult = crowdResult; // [BARU] update cache untuk frame-frame berikutnya
        }

        auto t1 = std::chrono::steady_clock::now();
        inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (_appConfig.isDetection) {
            detections = applyRuntimeFilter(rawDetections, runtimeAiConfig);

            std::vector<ApiDetection> apiDets;
            apiDets.reserve(detections.size());
            int idx = 0;
            for (const auto& det : detections) {
                ApiDetection ad;
                ad.id = "det_" + std::to_string(frameId) + "_" + std::to_string(idx++);
                ad.cls = detector->getClassName(det.classId);
                ad.confidence = det.confidence;
                ad.bbox_x = static_cast<double>(det.box.x) / frame.cols;
                ad.bbox_y = static_cast<double>(det.box.y) / frame.rows;
                ad.bbox_w = static_cast<double>(det.box.width)  / frame.cols;
                ad.bbox_h = static_cast<double>(det.box.height) / frame.rows;
                apiDets.push_back(ad);
            }
            apiController.broadcastDetections(frameId, frame.cols, frame.rows, apiDets, inferMs);

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

        // ---- Merge hasil Crowd Counting (heatmap + estimasi jumlah) ke frame ----
        if (_appConfig.isCrowdCounting && crowdCounter && crowdResult.valid) {
            cv::addWeighted(frame, 1.0, crowdResult.heatmapOverlay, 0.4, 0.0, frame);

            std::string crowdText = "Estimasi Kerumunan: " + std::to_string(crowdResult.estimatedCount);
            cv::putText(frame, crowdText, cv::Point(20, 160),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 255), 3);

            // apiController.broadcastCrowdCount(frameId, crowdResult.estimatedCount);
        }

        ++displayFrameCount;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - displayFpsWindowStart).count();

        if (elapsed >= 1.0) {
            displayFps = displayFrameCount / elapsed;
            displayFrameCount = 0;
            displayFpsWindowStart = now;

            AIConfig statusCfg;
            {
                std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);
                statusCfg.confidence_threshold = runtimeAiConfig.confidenceThreshold;
                for (int c : runtimeAiConfig.targetClasses) {
                    statusCfg.classes_enabled.insert(std::to_string(c));
                }
            }
            apiController.broadcastStatus(_appConfig.modelPath, displayFps, statusCfg, _appConfig.isDetection, _appConfig.isCrowdCounting);
        }

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