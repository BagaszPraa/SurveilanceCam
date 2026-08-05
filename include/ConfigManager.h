// ConfigManager.h
//
// Mengelola semua hal terkait pembacaan konfigurasi:
//   - Baca file config.ini (format key=value sederhana)
//   - Parse & override lewat argumen CLI
//   - Cetak ringkasan konfigurasi aktif ke log
//
// Dipisah dari main.cpp supaya main.cpp cuma fokus ke alur utama
// (capture -> infer -> stream), bukan detail parsing config.

#pragma once

#include <string>
#include <vector>
#include "rtspServer.h"
// ---------------- Konfigurasi Global ----------------
// Bisa diatur lewat file config.ini ATAU argumen CLI (CLI menang kalau ada dua-duanya).
struct Config {
    std::string inputSource;                      // index kamera ("0","1",...) atau path device (/dev/video0) atau path file video
    std::string modelPath;
    int width;                      // hanya "permintaan awal" (lihat catatan di atas)
    int height;                       // hanya "permintaan awal" (lihat catatan di atas)
    int fps;
    int inferSize;                       // imgsz model (samakan dengan saat export!)
    float confThresh;
    float nmsThresh;
    std::vector<int> targetClasses;                  // kosong = semua class
    int rtspPort;
    std::string rtspMount;
    std::string ipAdress;                // default IP address
    bool isGstreamer;   // true = input via pipeline GStreamer (mis. RTSP)
    EncoderType encoderType;
    CodecType codecType;
    int bitrateKbps;
    int reconnectIntervalMs;    // jeda antar percobaan reconnect saat input belum/tidak tersedia
    bool showOverlay;    // tampilkan teks info (Count/Infer/FPS/Resolusi) -- TIDAK memengaruhi bbox
    int apiPort;
    std::string apiHost;  // bind ke semua interface
    bool isDetection;     // aktifkan modul deteksi (YOLO)
    bool isCrowdCounting; // aktifkan modul crowd counting (heatmap + estimasi
    std::string crowdModelPath; // path ke model crowd counting
    int         crowdInputWidth     = 1024;
    int         crowdInputHeight    = 768;
    int         crowdInferInterval  = 10;   // 1 = tiap frame, N = tiap N frame
};
class ConfigManager {
public:
    // Alur lengkap, dipanggil dari main():
    //   1. Cek apakah ada --config <path> di argv (sebelum argumen lain diparse)
    //   2. Load dari file config.ini (kalau ada; kalau tidak ada, pakai default & tidak fatal)
    //   3. Override dengan argumen CLI (CLI selalu menang atas file config)
    //   4. Cetak ringkasan konfigurasi aktif ke log
    //
    // Contoh pemakaian di main.cpp:
    //   Config cfg = ConfigManager::load(argc, argv);
    // static Config load(int argc, char* argv[]);
    static Config load(int argc, char* argv[], std::string& outConfigPath);

    // Method di bawah ini dipecah terpisah (public) supaya bisa dipanggil
    // granular kalau perlu, misal untuk unit test atau reload config
    // tanpa restart program.
    static bool loadFromFile(const std::string& path, Config& cfg);
    static bool saveToFile(const std::string& path, const Config& cfg);
    static void applyCliArgs(int argc, char* argv[], Config& cfg);
    static void logSummary(const Config& cfg);

private:
    // ---- Helper parsing internal ----
    static std::string trim(const std::string& s);
    static std::vector<int> parseIntList(const std::string& value);
    static bool parseBool(const std::string& value);
    static EncoderType parseEncoderType(const std::string& value, EncoderType fallback);
    static CodecType parseCodecType(const std::string& value, CodecType fallback);
    static std::string encoderTypeToString(EncoderType type);
    static std::string codecTypeToString(CodecType type);
    // Cari nilai --config <path> di argv sebelum parsing argumen lain.
    // Default: "../config.ini" kalau tidak ada flag --config.
    static std::string findConfigPath(int argc, char* argv[]);

    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);

};