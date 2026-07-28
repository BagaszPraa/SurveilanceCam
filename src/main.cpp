// ==========================================================
// main.cpp
// yolo_local_test - Versi tanpa GStreamer, untuk testing lokal.
//   Input (USB webcam via OpenCV VideoCapture)
//   -> inferensi YOLO (TensorRT engine)
//   -> stream hasil via RTSP server
//
// Konfigurasi dibaca dari file eksternal (default: config.ini)
// supaya tidak perlu compile ulang tiap kali ganti setting.
// Urutan prioritas nilai konfigurasi:
//   default (hardcoded) < isi file config < argumen CLI
//
// CATATAN RESOLUSI (dinamis terhadap source):
//   cfg.width/cfg.height sekarang hanya dipakai sebagai "permintaan awal"
//   (mis. untuk cap.set() di kamera V4L2). Resolusi yang BENAR-BENAR dipakai
//   untuk RTSP server & seluruh pipeline ditentukan dari frame pertama yang
//   berhasil diambil dari source (probe frame). Jadi kalau source (kamera,
//   file video, atau RTSP input) punya resolusi native yang beda dari
//   config, program akan otomatis menyesuaikan -- tidak dipaksa resize ke
//   nilai config.
//
// CATATAN RETRY/RECONNECT (input belum tersedia / terputus):
//   Program TIDAK langsung berhenti kalau input belum tersedia saat start,
//   atau tiba-tiba terputus saat runtime (mis. USB webcam kecabut, RTSP
//   source mati sebentar). Sebagai gantinya program akan terus mencoba
//   menyambung ulang tiap `reconnect_interval_ms` (default 2000ms), dengan
//   log peringatan periodik (tidak setiap percobaan, supaya tidak spam),
//   dan RTSP server tetap hidup menunggu source kembali tersedia.
//
// CATATAN OVERLAY INFO (bukan bounding box):
//   `show_overlay` (config.ini) atau `--overlay`/`--no-overlay` (CLI)
//   mengatur tampil/tidaknya teks info (Count/Infer ms/Capture FPS/Display
//   FPS/Resolusi) di pojok kiri atas frame. Bounding box + label hasil
//   deteksi TIDAK terpengaruh oleh setting ini -- selalu digambar.
// ==========================================================

#include "main.h"

#include <opencv2/opencv.hpp>
#include "YoloDetector.h"
#include "rtspServer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cctype>

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

// ---------------- Helper: trim spasi di kiri/kanan string ----------------
std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// tambahkan kembali helper ini
bool isNetworkStream(const std::string& src) {
    return src.rfind("rtsp://", 0) == 0 ||
           src.rfind("rtmp://", 0) == 0 ||
           src.rfind("http://", 0) == 0 ||
           src.rfind("https://", 0) == 0;
}

// Pipeline GStreamer untuk RTSP input via OpenCV appsink.
// latency=0 supaya jitter buffer rtspsrc minim (low-latency).
// videoconvert + BGR karena OpenCV cv::Mat default-nya BGR.
// drop=true + max-buffers=1 di appsink supaya selalu ambil frame TERBARU,
// bukan antre (mirip pola "latest frame" yang sudah dipakai di capture thread).
std::string buildRtspGstPipeline(const std::string& url) {
    return
        "rtspsrc location=" + url + " latency=0 protocols=tcp "
        "! rtph264depay "
        "! h264parse "
        "! avdec_h264 "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false";
}

// ---------------- Helper: parse "1,2,3" -> {1,2,3} ----------------
std::vector<int> parseIntList(const std::string& value) {
    std::vector<int> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) result.push_back(std::stoi(item));
    }
    return result;
}

bool parseBool(const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "1" || v == "true" || v == "yes" || v == "on");
}

// ---------------- Baca file config bergaya INI sederhana ----------------
// Format:
//   # komentar
//   key = value
// Baris kosong dan yang diawali '#' atau ';' diabaikan.
bool loadConfigFile(const std::string& path, Config& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false; // file tidak ada -> pakai default, bukan error fatal
    }

    std::string line;
    int lineNo = 0;
    while (std::getline(file, line)) {
        ++lineNo;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#' || trimmed[0] == ';') continue;

        size_t eqPos = trimmed.find('=');
        if (eqPos == std::string::npos) {
            logWarn("Baris " + std::to_string(lineNo) + " diabaikan (format salah): " + line);
            continue;
        }

        std::string key   = trim(trimmed.substr(0, eqPos));
        std::string value = trim(trimmed.substr(eqPos + 1));
        if (value.empty()) continue;

        try {
            if (key == "input")               cfg.inputSource  = value;
            else if (key == "model")          cfg.modelPath    = value;
            else if (key == "width")          cfg.width        = std::stoi(value);
            else if (key == "height")         cfg.height       = std::stoi(value);
            else if (key == "fps")            cfg.fps          = std::stoi(value);
            else if (key == "infer_size")     cfg.inferSize    = std::stoi(value);
            else if (key == "conf")           cfg.confThresh   = std::stof(value);
            else if (key == "nms")            cfg.nmsThresh    = std::stof(value);
            else if (key == "target_classes") cfg.targetClasses = parseIntList(value);
            else if (key == "rtsp_port")      cfg.rtspPort     = std::stoi(value);
            else if (key == "rtsp_mount")     cfg.rtspMount    = value;
            else if (key == "ip_address")     cfg.ipAdress     = value;
            else if (key == "gstreamer")      cfg.isGstreamer  = parseBool(value);
            else if (key == "reconnect_interval_ms") cfg.reconnectIntervalMs = std::stoi(value);
            else if (key == "show_overlay")   cfg.showOverlay  = parseBool(value);
            else logWarn("Key tidak dikenal di baris " + std::to_string(lineNo) + ": " + key);
        } catch (const std::exception& e) {
            logWarn("Gagal parsing baris " + std::to_string(lineNo) + " (" + key + "=" + value + "): " + e.what());
        }
    }
    return true;
}

// ---------------- Helper: buka/ulang-buka VideoCapture sesuai konfigurasi ----------------
// Dipakai baik untuk membuka input pertama kali maupun untuk reconnect
// ketika input hilang/terputus di tengah jalan. Mengembalikan false
// (bukan exit program) kalau gagal, supaya pemanggil bisa retry.
bool openCapture(const Config& cfg, cv::VideoCapture& cap) {
    bool isNumericIndex = !cfg.inputSource.empty() &&
        std::all_of(cfg.inputSource.begin(), cfg.inputSource.end(), ::isdigit);

    if (cfg.isGstreamer) {
        std::string gstPipeline = isNetworkStream(cfg.inputSource)
            ? buildRtspGstPipeline(cfg.inputSource)   // user cuma kasih URL polos -> auto-build pipeline
            : cfg.inputSource;                        // user sudah kasih pipeline GStreamer lengkap -> pakai apa adanya
        cap.open(gstPipeline, cv::CAP_GSTREAMER);
    } else if (isNumericIndex) {
        // USB webcam via index
        cap.open(std::stoi(cfg.inputSource), cv::CAP_V4L2);
    } else {
        // path device (/dev/video0) atau file video lokal
        cap.open(cfg.inputSource, cv::CAP_V4L2);
    }

    if (!cap.isOpened()) {
        return false;
    }

    if (!cfg.isGstreamer) {
        // Properti ini cuma relevan untuk device kamera lokal (V4L2).
        // Ini hanya "permintaan" ke driver -- resolusi FINAL tetap dideteksi
        // ulang dari frame asli (probe frame), karena driver/kamera bisa
        // saja tidak persis menuruti angka yang diminta.
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
        cap.set(cv::CAP_PROP_FPS, cfg.fps);
    }

    return true;
}

int main(int argc, char* argv[]) {
    Config cfg;

    // ---- Path config file bisa di-override lewat --config, sebelum argumen lain diparse ----
    std::string configPath = "../config.ini";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            configPath = argv[i + 1];
            break;
        }
    }

    if (loadConfigFile(configPath, cfg)) {
        logInfo("Berhasil memuat konfigurasi dari: " + configPath);
    } else {
        logInfo("File config '" + configPath +
                "' tidak ditemukan, memakai nilai default (bisa juga diatur lewat argumen CLI).");
    }

    // ---- Parsing argumen CLI (override nilai dari file config bila diberikan) ----
    // Contoh pemakaian:
    //   ./yolo_local_test --config myconfig.ini
    //   ./yolo_local_test --input 0 --model ../models/yolo11s.engine
    //   ./yolo_local_test --input /dev/video0 --model ../models/visDrone.engine --infer-size 832
    //   ./yolo_local_test --no-overlay
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextVal = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
        };

        if (arg == "--config")          nextVal("");   // sudah ditangani di atas, lewati saja nilainya
        else if (arg == "--input")      cfg.inputSource = nextVal(cfg.inputSource.c_str());
        else if (arg == "--model")      cfg.modelPath   = nextVal(cfg.modelPath.c_str());
        else if (arg == "--width")      cfg.width       = std::stoi(nextVal(std::to_string(cfg.width).c_str()));
        else if (arg == "--height")     cfg.height      = std::stoi(nextVal(std::to_string(cfg.height).c_str()));
        else if (arg == "--fps")        cfg.fps         = std::stoi(nextVal(std::to_string(cfg.fps).c_str()));
        else if (arg == "--infer-size") cfg.inferSize   = std::stoi(nextVal(std::to_string(cfg.inferSize).c_str()));
        else if (arg == "--conf")       cfg.confThresh  = std::stof(nextVal(std::to_string(cfg.confThresh).c_str()));
        else if (arg == "--nms")        cfg.nmsThresh   = std::stof(nextVal(std::to_string(cfg.nmsThresh).c_str()));
        else if (arg == "--rtsp-port")  cfg.rtspPort    = std::stoi(nextVal(std::to_string(cfg.rtspPort).c_str()));
        else if (arg == "--rtsp-mount") cfg.rtspMount   = nextVal(cfg.rtspMount.c_str());
        else if (arg == "--gstreamer")  cfg.isGstreamer = true;   // <-- tambahan, tanpa nilai (flag saklar)
        else if (arg == "--no-gstreamer") cfg.isGstreamer = false; // opsional, buat override balik dari config
        else if (arg == "--reconnect-interval") cfg.reconnectIntervalMs = std::stoi(nextVal(std::to_string(cfg.reconnectIntervalMs).c_str()));
        else if (arg == "--overlay")    cfg.showOverlay = true;    // tampilkan teks info
        else if (arg == "--no-overlay") cfg.showOverlay = false;   // sembunyikan teks info (bbox tetap tampil)
    }

    logInfo("==========================================");
    logInfo("yolo_local_test - konfigurasi aktif");
    logInfo("Input      : " + cfg.inputSource + " (gstreamer=" + (cfg.isGstreamer ? "true" : "false") + ")");
    logInfo("Model      : " + cfg.modelPath);
    logInfo("Resolusi   : " + std::to_string(cfg.width) + "x" + std::to_string(cfg.height) + " @ " +
            std::to_string(cfg.fps) + "fps (permintaan awal, resolusi final mengikuti source)");
    logInfo("InferSize  : " + std::to_string(cfg.inferSize));
    logInfo("Conf/NMS   : " + std::to_string(cfg.confThresh) + " / " + std::to_string(cfg.nmsThresh));
    logInfo("RTSP       : port " + std::to_string(cfg.rtspPort) + ", mount " + cfg.rtspMount);
    logInfo("Reconnect  : tiap " + std::to_string(cfg.reconnectIntervalMs) + " ms jika input belum/tidak tersedia");
    logInfo(std::string("Overlay    : ") + (cfg.showOverlay ? "aktif" : "nonaktif") +
            " (teks info Count/Infer/FPS/Resolusi; bbox tetap selalu digambar)");
    logInfo("==========================================");

    // ---- Buka input via OpenCV VideoCapture (retry sampai berhasil, TIDAK exit) ----
    // Kalau inputSource cuma angka ("0","1",...), buka sebagai index kamera.
    // Kalau bukan angka (path device "/dev/video0" atau file video), buka sebagai string.
    cv::VideoCapture cap;
    {
        int attempt = 0;
        while (true) {
            ++attempt;
            if (openCapture(cfg, cap)) {
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

    logInfo("Kamera/berkas terbuka.");

    // ---- PROBE FRAME: deteksi resolusi asli dari source (retry sampai berhasil, TIDAK exit) ----
    // Ambil frame pertama sampai berhasil (kadang frame pertama dari kamera/
    // stream masih kosong/gagal), lalu pakai ukuran frame itu sebagai
    // resolusi ACUAN untuk seluruh pipeline (bukan cfg.width/height yang
    // cuma "permintaan"). Kalau koneksi ternyata putus di tahap ini, program
    // akan mencoba membuka ulang input alih-alih berhenti.
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

            // Kalau capture sudah tidak valid lagi (device kecabut dsb),
            // coba buka ulang dari awal.
            if (!cap.isOpened()) {
                logWarn("Koneksi ke input terputus saat probe, mencoba membuka ulang...");
                cap.release();
                int reopenAttempt = 0;
                while (!openCapture(cfg, cap)) {
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
    cv::Mat latestCapturedFrame = probeFrame; // langsung isi dengan probe frame supaya tidak terbuang
    std::atomic<bool> captureRunning{true};
    std::atomic<bool> frameAvailable{true};   // sudah true karena probe frame sudah ada
    std::mutex fpsMutex;
    double captureFps = 0.0;

    // ---- Thread capture terpisah ----
    // cap.read() dijalankan terus-menerus secepat mungkin di background,
    // menimpa latestCapturedFrame tiap dapat frame baru. Ini memastikan
    // main thread (infer+push) selalu pakai frame TERBARU, bukan frame
    // lama yang ngantre di buffer -- tanpa perlu blocking/menunggu apapun.
    //
    // Kalau cap.read() gagal (device kecabut/putus), thread ini TIDAK
    // menghentikan program. Sebagai gantinya ia mencoba menyambung ulang
    // (release + openCapture) terus-menerus sampai berhasil, sambil
    // menandai frameAvailable=false supaya main loop cuma menunggu tanpa
    // mendorong frame lama/rusak ke RTSP server.
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
                while (captureRunning && !openCapture(cfg, cap)) {
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

            // Pastikan 3 channel BGR (bukan BGRA/grayscale)
            if (tmp.channels() == 4) {
                cv::cvtColor(tmp, tmp, cv::COLOR_BGRA2BGR);
            }

            // Resolusi acuan (cfg.width/height) sudah ditentukan dari probe
            // frame di atas, jadi mengikuti source apa adanya. Resize di sini
            // HANYA dipakai sebagai jaring pengaman kalau source berubah
            // resolusi di tengah jalan (jarang terjadi), supaya caps RTSP
            // yang sudah terlanjur dibuat tetap konsisten.
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
        // Tunggu sampai frame tersedia (frameAvailable bisa jadi false
        // sementara kalau capture thread sedang reconnect), lalu ambil
        // frame TERBARU yang ada saat ini (bukan antre menunggu giliran
        // seperti queue biasa).
        if (!frameAvailable) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(captureMutex);
            if (!frameAvailable) continue; // double check setelah lock, bisa berubah barusan
            frame = latestCapturedFrame.clone(); // clone supaya aman dipakai di luar lock
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<Detection> detections = detector->infer(frame);
        auto t1 = std::chrono::steady_clock::now();
        double inferMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // ---- Bounding box + label hasil deteksi (SELALU digambar, tidak
        //      dipengaruhi cfg.showOverlay -- itu cuma untuk teks info) ----
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

        // ---- Hitung Display FPS (rate loop utama proses+push) ----
        ++displayFrameCount;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - displayFpsWindowStart).count();
        if (elapsed >= 1.0) {
            displayFps = displayFrameCount / elapsed;
            displayFrameCount = 0;
            displayFpsWindowStart = now;
        }

        // ---- Info overlay (teks Count/Infer/FPS/Resolusi) -- bisa
        //      dimatikan lewat cfg.showOverlay, TIDAK memengaruhi bbox di atas ----
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