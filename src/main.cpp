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

#include "YoloDetector.h"
#include "rtspServer.h"
#include "ConfigManager.h"
#include "APIController.h"
#include "SourceHandler.h"

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

// ---------------- Runtime AI config, diubah live via APIController ----------------
struct RuntimeAIConfig {
    std::mutex mtx;
    float confidenceThreshold;
    std::set<int> targetClasses;   // kosong = semua class diizinkan
};

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
    Config cfg = ConfigManager::load(argc, argv);
    SourceHandler sourceHandler;
    cv::VideoCapture cap;
    {
        int attempt = 0;
        while (true) {
            ++attempt;
            if (sourceHandler.openCapture(cfg, cap)) {
                logInfo("Input berhasil dibuka pada percobaan ke-" + std::to_string(attempt));
                break;
            }

            if (attempt == 1 || attempt % 10 == 0) {
                logWarn("Input '" + cfg.inputSource + "' belum tersedia (percobaan ke-" +
                        std::to_string(attempt) + "). Mencoba lagi tiap " +
                        std::to_string(cfg.reconnectIntervalMs) + " ms...");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.reconnectIntervalMs));
        }
    }

    // CATATAN: BUFFERSIZE sengaja TIDAK dipaksa ke 1 di sini. Kernel V4L2 perlu
    // beberapa buffer supaya bisa terus capture frame baru sementara userspace
    // masih sibuk decode JPEG frame sebelumnya (overlap capture+decode).
    // Kalau buffer cuma 1, kernel harus nunggu kita "lepas" buffer itu dulu
    // (selesai decode) sebelum bisa capture frame berikutnya -- capture jadi
    // serial dengan decode, bukan paralel, dan FPS efektif turun drastis.
    // Mekanisme "selalu ambil frame terbaru" sudah kita tangani sendiri lewat
    // thread capture terpisah + overwrite di bawah, jadi buffer besar di sini aman.
    if (!cfg.isGstreamer) {
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
            cfg.modelPath, cfg.inferSize, cfg.confThresh, cfg.nmsThresh, cfg.targetClasses);
    } catch (const std::exception& e) {
        logError("Gagal load model: " + cfg.modelPath + " | Pesan: " + e.what());
        return 1;
    }

    // ---- Runtime AI config (thread-safe), state awal dari cfg ----
    RuntimeAIConfig runtimeAiConfig;
    runtimeAiConfig.confidenceThreshold = cfg.confThresh;
    runtimeAiConfig.targetClasses = std::set<int>(cfg.targetClasses.begin(), cfg.targetClasses.end());

    // ---- APIController: WebSocket server untuk GCS ----
    APIController apiController;

    apiController.setConfigCommandHandler([&](const AIConfig& newCfg) -> bool {
        std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);
        runtimeAiConfig.confidenceThreshold = static_cast<float>(newCfg.confidence_threshold);

        // Konvensi: classes_enabled dari GCS berisi class ID dalam bentuk
        // string, mis. {"0", "2"} -- konsisten dengan format cfg.targetClasses
        // di config.ini (comma-separated int).
        runtimeAiConfig.targetClasses.clear();
        for (const auto& s : newCfg.classes_enabled) {
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

    // run() itu blocking (io_service.run()), jadi wajib di thread terpisah.
    // detach() dipakai di sini karena APIController belum punya mekanisme
    // stop() yang graceful -- kalau perlu shutdown bersih, tambahkan method
    // stop() yang panggil server_.stop_listening() + server_.stop().
    std::thread apiThread([&apiController, &cfg]() {
        apiController.run(static_cast<uint16_t>(cfg.apiPort));
    });
    apiThread.detach();

    // logInfo("APIController berjalan di port " + std::to_string(cfg.apiPort));
    // logInfo("Kamera/berkas terbuka.");

    // ---- PROBE FRAME: deteksi resolusi asli dari source (retry sampai berhasil, TIDAK exit) ----
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
                while (!sourceHandler.openCapture(cfg, cap)) {
                    ++reopenAttempt;
                    if (reopenAttempt == 1 || reopenAttempt % 10 == 0) {
                        logWarn("Percobaan buka ulang ke-" + std::to_string(reopenAttempt) +
                                " gagal, mencoba lagi tiap " + std::to_string(cfg.reconnectIntervalMs) + " ms...");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.reconnectIntervalMs));
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

    if (actualWidth != cfg.width || actualHeight != cfg.height) {
        logInfo("Resolusi asli dari source: " + std::to_string(actualWidth) + "x" + std::to_string(actualHeight) +
                " (berbeda dari config " + std::to_string(cfg.width) + "x" + std::to_string(cfg.height) +
                "). Memakai resolusi asli source secara dinamis.");
    } else {
        logInfo("Resolusi asli dari source: " + std::to_string(actualWidth) + "x" + std::to_string(actualHeight));
    }
    cfg.width  = actualWidth;
    cfg.height = actualHeight;

    RtspServer rtspServer(
        cfg.rtspPort,
        cfg.rtspMount,
        cfg.width,
        cfg.height,
        cfg.fps,
        cfg.ipAdress,
        cfg.encoderType,
        cfg.codecType,
        cfg.bitrateKbps
    );

    if (!rtspServer.start()) {
        logError("Gagal menjalankan RTSP Server.");
        return 1;
    }

    // ---- State frame terbaru, dipakai bareng antara capture thread & main thread ----
    std::mutex captureMutex;
    cv::Mat latestCapturedFrame = probeFrame;
    std::atomic<bool> captureRunning{true};
    std::atomic<bool> frameAvailable{true};
    std::mutex fpsMutex;
    double captureFps = 0.0;

    // ---- Thread capture terpisah ----
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
                while (captureRunning && !sourceHandler.openCapture(cfg, cap)) {
                    ++attempt;
                    if (attempt == 1 || attempt % 10 == 0) {
                        logWarn("Reconnect percobaan ke-" + std::to_string(attempt) +
                                " gagal, mencoba lagi tiap " + std::to_string(cfg.reconnectIntervalMs) + " ms...");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(cfg.reconnectIntervalMs));
                }

                if (!captureRunning) break;

                logInfo("Berhasil menyambung ulang ke source, melanjutkan streaming.");
                resolutionMismatchWarned = false;
                continue;
            }

            if (tmp.channels() == 4) {
                cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
            }

            if (tmp.cols != cfg.width || tmp.rows != cfg.height) {
                if (!resolutionMismatchWarned) {
                    logWarn("Resolusi frame berubah jadi " + std::to_string(tmp.cols) + "x" + std::to_string(tmp.rows) +
                            " (acuan " + std::to_string(cfg.width) + "x" + std::to_string(cfg.height) +
                            "). Melakukan resize agar stream RTSP tetap konsisten.");
                    resolutionMismatchWarned = true;
                }
                cv::resize(tmp, tmp, cv::Size(cfg.width, cfg.height));
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

        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> rawDetections = detector->infer(frame);
        auto t1 = std::chrono::steady_clock::now();
        double inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Terapkan threshold/class filter yang bisa diubah live lewat APIController.
        std::vector<Detection> detections = applyRuntimeFilter(rawDetections, runtimeAiConfig);

        // ---- Broadcast hasil deteksi ke GCS via APIController ----
        {
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
            ++frameId;
        }

        // ---- Bounding box + label hasil deteksi ----
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

            std::string label =
                std::string(detector->getClassName(det.classId)) +
                " " +
                cv::format("%.2f", det.confidence);

            cv::putText(frame,
                        label,
                        cv::Point(det.box.x, det.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 0),
                        2);
        }

        // ---- Hitung Display FPS ----
        ++displayFrameCount;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - displayFpsWindowStart).count();

        if (elapsed >= 1.0) {
            displayFps = displayFrameCount / elapsed;
            displayFrameCount = 0;
            displayFpsWindowStart = now;

            // ---- Broadcast status AI engine ke GCS ----
            AIConfig statusCfg;
            {
                std::lock_guard<std::mutex> lock(runtimeAiConfig.mtx);
                statusCfg.confidence_threshold = runtimeAiConfig.confidenceThreshold;
                for (int c : runtimeAiConfig.targetClasses) {
                    statusCfg.classes_enabled.insert(std::to_string(c));
                }
            }
            apiController.broadcastStatus(cfg.modelPath, displayFps, statusCfg);
        }

        // ---- Info overlay ----
        if (cfg.showOverlay) {
            double captureFpsSnapshot;
            {
                std::lock_guard<std::mutex> fpsLock(fpsMutex);
                captureFpsSnapshot = captureFps;
            }
            std::string countText = "Count: " + std::to_string(detections.size()) +
                                     "  |  Infer: " + std::to_string(static_cast<int>(inferMs)) + " ms";
            std::string fpsText = "Capture FPS: " + std::to_string(static_cast<int>(captureFpsSnapshot)) +
                                   "  |  Display FPS: " + std::to_string(static_cast<int>(displayFps));
            std::string resText = "Resolusi: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows);

            cv::putText(frame, countText, cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
            cv::putText(frame, fpsText, cv::Point(20, 80),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
            cv::putText(frame, resText, cv::Point(20, 120),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 1);
        }

        rtspServer.pushFrame(frame);
    }

    captureRunning = false;
    if (captureThread.joinable()) captureThread.join();

    cap.release();

    logInfo("Program selesai.");
    return 0;
}