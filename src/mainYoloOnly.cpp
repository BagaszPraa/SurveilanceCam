// ==========================================================
// yolo_local_test
// Versi tanpa GStreamer, untuk testing lokal.
//   Input (USB webcam via OpenCV VideoCapture)
//   -> inferensi YOLO (TensorRT engine)
//   -> tampilkan hasil di window OpenCV (cv::imshow)
//
// Tekan 'q' atau ESC di window untuk keluar.
// ==========================================================

#include <opencv2/opencv.hpp>
#include "YoloDetector.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cctype>

// ---------------- Konfigurasi Global (bisa via argumen CLI) ----------------
struct Config {
    std::string inputSource = "0";                    // index kamera ("0","1",...) atau path device (/dev/video0) atau path file video
    std::string modelPath   = "../models/yolo11s.engine";
    int width               = 1280;
    int height              = 720;
    int inferSize           = 640;                     // imgsz model (samakan dengan saat export!)
    float confThresh        = 0.25f;
    std::vector<int> targetClasses = {};                // kosong = semua class
};

int main(int argc, char* argv[]) {
    Config cfg;

    // ---- Parsing argumen sederhana ----
    // Contoh pemakaian:
    //   ./yolo_local_test --input 0 --model ../models/yolo11s.engine
    //   ./yolo_local_test --input /dev/video0 --model ../models/visDrone.engine --infer-size 832
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextVal = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
        };

        if (arg == "--input")           cfg.inputSource = nextVal(cfg.inputSource.c_str());
        else if (arg == "--model")      cfg.modelPath   = nextVal(cfg.modelPath.c_str());
        else if (arg == "--width")      cfg.width       = std::stoi(nextVal("1280"));
        else if (arg == "--height")     cfg.height      = std::stoi(nextVal("720"));
        else if (arg == "--infer-size") cfg.inferSize   = std::stoi(nextVal("640"));
        else if (arg == "--conf")       cfg.confThresh  = std::stof(nextVal("0.25"));
    }

    std::cout << "==========================================\n";
    std::cout << " yolo_local_test - konfigurasi\n";
    std::cout << " Input      : " << cfg.inputSource << "\n";
    std::cout << " Model      : " << cfg.modelPath << "\n";
    std::cout << " Resolusi   : " << cfg.width << "x" << cfg.height << "\n";
    std::cout << " InferSize  : " << cfg.inferSize << "\n";
    std::cout << " Conf       : " << cfg.confThresh << "\n";
    std::cout << "==========================================\n";

    // ---- Buka input via OpenCV VideoCapture ----
    // Kalau inputSource cuma angka ("0","1",...), buka sebagai index kamera.
    // Kalau bukan angka (path device "/dev/video0" atau file video), buka sebagai string.
    cv::VideoCapture cap;
    bool isNumericIndex = !cfg.inputSource.empty() &&
        std::all_of(cfg.inputSource.begin(), cfg.inputSource.end(), ::isdigit);

    if (isNumericIndex) {
        cap.open(std::stoi(cfg.inputSource), cv::CAP_V4L2);
    } else {
        cap.open(cfg.inputSource, cv::CAP_V4L2);
    }

    if (!cap.isOpened()) {
        std::cerr << "[Capture] Gagal buka input: " << cfg.inputSource << std::endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    cap.set(cv::CAP_PROP_FPS, 30);
    // CATATAN: BUFFERSIZE sengaja TIDAK dipaksa ke 1 di sini. Kernel V4L2 perlu
    // beberapa buffer supaya bisa terus capture frame baru sementara userspace
    // masih sibuk decode JPEG frame sebelumnya (overlap capture+decode).
    // Kalau buffer cuma 1, kernel harus nunggu kita "lepas" buffer itu dulu
    // (selesai decode) sebelum bisa capture frame berikutnya -- capture jadi
    // serial dengan decode, bukan paralel, dan FPS efektif turun drastis.
    // Mekanisme "selalu ambil frame terbaru" sudah kita tangani sendiri lewat
    // thread capture terpisah + overwrite di bawah, jadi buffer besar di sini aman.

    std::cout << "[Capture] FOURCC aktual: ";
    {
        int fourccInt = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
        char fourccStr[5] = {
            static_cast<char>(fourccInt & 0xFF),
            static_cast<char>((fourccInt >> 8) & 0xFF),
            static_cast<char>((fourccInt >> 16) & 0xFF),
            static_cast<char>((fourccInt >> 24) & 0xFF),
            '\0'
        };
        std::cout << fourccStr << " | FPS diminta ke driver: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    }

    // ---- Load model YOLO ----
    std::unique_ptr<YoloDetector> detector;
    try {
        detector = std::make_unique<YoloDetector>(
            cfg.modelPath, cfg.inferSize, cfg.confThresh, 0.45f, cfg.targetClasses);
    } catch (const std::exception& e) {
        std::cerr << "[Capture] Gagal load model: " << cfg.modelPath << "\n"
                  << "          Pesan: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "[Capture] Kamera/berkas terbuka. Tekan 'q' atau ESC untuk keluar." << std::endl;

    const std::string windowName = "YOLO Detection - Local Test";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);

    // ---- Thread capture terpisah ----
    // cap.read() dijalankan terus-menerus secepat mungkin di background,
    // menimpa g_latestCapturedFrame tiap dapat frame baru. Ini memastikan
    // main thread (infer+display) selalu pakai frame TERBARU, bukan frame
    // lama yang ngantre di buffer -- tanpa perlu blocking/menunggu apapun.
    std::mutex captureMutex;
    cv::Mat latestCapturedFrame;
    std::atomic<bool> captureRunning{true};
    std::atomic<bool> frameAvailable{false};
    std::mutex fpsMutex;
    double captureFps = 0.0;

    std::thread captureThread([&]() {
        cv::Mat tmp;
        int frameCount = 0;
        auto fpsWindowStart = std::chrono::steady_clock::now();

        while (captureRunning) {
            if (!cap.read(tmp) || tmp.empty()) {
                std::cerr << "[Capture] Gagal ambil frame dari kamera." << std::endl;
                captureRunning = false;
                break;
            }
            {
                std::lock_guard<std::mutex> lock(captureMutex);
                latestCapturedFrame = tmp; // cv::Mat assignment = shallow copy (murah)
                frameAvailable = true;
            }

            // ---- Hitung Capture FPS (rate asli kamera kirim frame) ----
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

    while (captureRunning) {
        // Tunggu sampai frame pertama tersedia, lalu ambil frame TERBARU
        // yang ada saat ini (bukan antre menunggu giliran seperti queue biasa)
        if (!frameAvailable) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(captureMutex);
            frame = latestCapturedFrame.clone(); // clone supaya aman dipakai di luar lock
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> detections = detector->infer(frame);
        auto t1 = std::chrono::steady_clock::now();
        double inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Gambar bbox ----
        for (const auto& det : detections) {
            cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);
        }

        // ---- Hitung Display FPS (rate loop utama proses+tampilkan) ----
        ++displayFrameCount;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - displayFpsWindowStart).count();
        if (elapsed >= 1.0) {
            displayFps = displayFrameCount / elapsed;
            displayFrameCount = 0;
            displayFpsWindowStart = now;
        }

        // ---- Info overlay ----
        std::string countText = "Count: " + std::to_string(detections.size()) +
                                 "  |  Infer: " + std::to_string(static_cast<int>(inferMs)) + " ms";
        double captureFpsSnapshot;
        {
            std::lock_guard<std::mutex> fpsLock(fpsMutex);
            captureFpsSnapshot = captureFps;
        }
        std::string fpsText = "Capture FPS: " + std::to_string(static_cast<int>(captureFpsSnapshot)) +
                               "  |  Display FPS: " + std::to_string(static_cast<int>(displayFps));

        cv::putText(frame, countText, cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, fpsText, cv::Point(20, 80),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow(windowName, frame);

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' atau ESC
            break;
        }
    }

    captureRunning = false;
    if (captureThread.joinable()) captureThread.join();

    cap.release();
    cv::destroyAllWindows();

    std::cout << "Program selesai." << std::endl;
    return 0;
}