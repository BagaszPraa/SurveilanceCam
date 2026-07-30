// ==========================================================
// main.h
// Deklarasi Config dan helper untuk yolo_local_test.
//
// CATATAN RESOLUSI (dinamis terhadap source):
//   cfg.width/cfg.height hanya dipakai sebagai "permintaan awal"
//   (mis. untuk cap.set() di kamera V4L2). Resolusi yang BENAR-BENAR
//   dipakai untuk RTSP server & seluruh pipeline ditentukan dari frame
//   pertama yang berhasil diambil dari source (probe frame) di main.cpp.
//
// CATATAN RETRY/RECONNECT (input belum tersedia / terputus):
//   Program TIDAK langsung berhenti kalau input belum tersedia saat start,
//   atau tiba-tiba terputus saat runtime (mis. USB webcam kecabut, RTSP
//   source mati sebentar). Program akan terus mencoba menyambung ulang
//   tiap `reconnectIntervalMs` (default 2000ms).
//
// CATATAN OVERLAY INFO (bukan bounding box):
//   cfg.showOverlay mengatur tampil/tidaknya teks info di pojok kiri atas
//   frame (Count/Infer ms/Capture FPS/Display FPS/Resolusi). Ini TIDAK
//   memengaruhi bounding box + label hasil deteksi -- bbox selalu
//   digambar terlepas dari nilai cfg.showOverlay ini.
// ==========================================================

#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "YoloDetector.h"
#include "rtspServer.h"
#include "APIController.h"

// ---------------- Konfigurasi Global ----------------
// Bisa diatur lewat file config.ini ATAU argumen CLI (CLI menang kalau ada dua-duanya).
struct Config {
    std::string inputSource = "0";                      // index kamera ("0","1",...) atau path device (/dev/video0) atau path file video
    std::string modelPath   = "../models/yolo11s.engine";
    int width               = 1280;                      // hanya "permintaan awal" (lihat catatan di atas)
    int height               = 720;                       // hanya "permintaan awal" (lihat catatan di atas)
    int fps                 = 30;
    int inferSize           = 640;                       // imgsz model (samakan dengan saat export!)
    float confThresh        = 0.25f;
    float nmsThresh         = 0.45f;
    std::vector<int> targetClasses = {};                  // kosong = semua class
    int rtspPort            = 8554;
    std::string rtspMount   = "/live";
    std::string ipAdress    = "127.0.0.1";                // default IP address
    bool isGstreamer        = false;   // true = input via pipeline GStreamer (mis. RTSP)
    EncoderType encoderType = EncoderType::NVIDIA_GPU;
    CodecType codecType     = CodecType::H265;
    int bitrateKbps         = 2000;
    int reconnectIntervalMs = 2000;    // jeda antar percobaan reconnect saat input belum/tidak tersedia
    bool showOverlay        = true;    // tampilkan teks info (Count/Infer/FPS/Resolusi) -- TIDAK memengaruhi bbox
    // --- Tambahan untuk APIController ---
    int apiPort             = 8765;
    std::string apiHost     = "0.0.0.0";  // bind ke semua interface
};

// ---------------- Helper: trim spasi di kiri/kanan string ----------------
std::string trim(const std::string& s);

// ---------------- Helper: deteksi apakah string berupa URL stream jaringan ----------------
bool isNetworkStream(const std::string& src);

// ---------------- Helper: bangun pipeline GStreamer untuk RTSP input via appsink ----------------
std::string buildRtspGstPipeline(const std::string& url);

// ---------------- Helper: parse "1,2,3" -> {1,2,3} ----------------
std::vector<int> parseIntList(const std::string& value);

// ---------------- Helper: parse string boolean ala INI ("1"/"true"/"yes"/"on") ----------------
bool parseBool(const std::string& value);

// ---------------- Baca file config bergaya INI sederhana ----------------
// Format:
//   # komentar
//   key = value
// Baris kosong dan yang diawali '#' atau ';' diabaikan.
// Mengembalikan false (bukan error fatal) kalau file tidak ditemukan --
// pemanggil tetap lanjut memakai nilai default di cfg.
bool loadConfigFile(const std::string& path, Config& cfg);

// ---------------- Helper: buka/ulang-buka VideoCapture sesuai konfigurasi ----------------
// Dipakai baik untuk membuka input pertama kali maupun untuk reconnect
// ketika input hilang/terputus di tengah jalan. Mengembalikan false
// (bukan exit program) kalau gagal, supaya pemanggil bisa retry.
bool openCapture(const Config& cfg, cv::VideoCapture& cap);

// ---------------- Logging, prefix [Main] ----------------
void logInfo(const std::string& msg);
void logWarn(const std::string& msg);
void logError(const std::string& msg);