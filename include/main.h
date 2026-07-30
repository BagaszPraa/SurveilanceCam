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

// ---------------- Logging, prefix [Main] ----------------
void logInfo(const std::string& msg);
void logWarn(const std::string& msg);
void logError(const std::string& msg);