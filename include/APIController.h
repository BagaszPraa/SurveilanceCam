// APIController.h
//
// WebSocket server sisi Drone/Edge (companion computer: Jetson/RPi).
// Bertugas:
//   - Broadcast hasil deteksi AI (bbox, class, confidence) ke GCS
//   - Terima command konfigurasi AI dari GCS (threshold, class filter, dll)
//
// Dependency: websocketpp (header-only) + nlohmann/json + Boost.Asio
//   sudo apt install libboost-system-dev nlohmann-json3-dev
//   git clone https://github.com/zaphoyd/websocketpp
//
// Kompilasi contoh:
//   g++ -std=c++17 -Iwebsocketpp APIController.cpp main_server.cpp \
//       -lboost_system -lpthread -o ai_server

#pragma once

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <nlohmann/json.hpp>
#include "ConfigManager.h"

#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;
using WsServer = websocketpp::server<websocketpp::config::asio>;
using ConnHandle = websocketpp::connection_hdl;

// ------------------------------------------------------------------
// Struct hasil deteksi tunggal
// ------------------------------------------------------------------
struct ApiDetectionResult {
    int id;
    std::string cls;         // "person", "vehicle", dll
    double confidence;
    double bbox_x, bbox_y, bbox_w, bbox_h;  // format: xywh, normalized 0..1
};

// ------------------------------------------------------------------
// Struct hasil crowd counting untuk satu frame.
// ------------------------------------------------------------------
struct ApiCrowdResult {
    long count = 0;
    int densityLevel;   // 0 = emboh | 1=normal | 2=medium | 3=crowded
};


// ------------------------------------------------------------------
// Struct konfigurasi AI yang bisa diubah lewat command "cmd" dari GCS.
//
// Nama member C++ di sini TIDAK selalu sama dengan key JSON di wire
// protocol -- lihat mapping lengkap di kFieldKey* (APIController.cpp)
// dan tabel di bawah. Ini karena beberapa key JSON pendek ("class")
// bentrok dengan keyword C++.
//
// Semua field bertipe std::optional supaya bisa dibedakan antara
// "tidak dikirim di command ini" vs "dikirim dengan nilai tertentu" --
// field yang tidak dikirim TIDAK menimpa nilai yang sedang berjalan.
//
// Golongan & mapping JSON key -> member:
//   [USER]      conf            -> confidenceThreshold   (real-time)
//               nms             -> iouThreshold           (real-time)
//               class           -> classesEnabled         (real-time)
//               detect          -> detectionEnabled       (real-time)
//               crowd           -> crowdCountingEnabled   (real-time)
//               overlay         -> showOverlay            (real-time)
//
//   [SEMI-DEV]  crowd_interval  -> crowdInferInterval     (butuh restart*)
//               crowd_width     -> crowdInputWidth        (butuh restart*)
//               crowd_height    -> crowdInputHeight       (butuh restart*)
//               infer_size      -> inferSize              (butuh restart*)
//               bitrate_kbps    -> bitrateKbps             (butuh restart*)
//               reconnect_ms    -> reconnectIntervalMs    (real-time**)
//
//   * Butuh restart aplikasi karena nilai ini dipakai untuk alokasi
//     buffer/engine saat startup (CrowdCounting/YoloDetector/RtspServer
//     encoder), bukan dibaca ulang tiap iterasi loop.
//   ** reconnectIntervalMs dibaca langsung dari Config master di setiap
//      iterasi reconnect loop, jadi bisa berlaku live tanpa restart.
//
//   [FULL-DEV]  ip_address, api_port, api_host
//               -> SENGAJA TIDAK ADA di struct ini / ditolak total kalau
//               muncul di params (lihat kFullDevOnlyKeys di .cpp).
// ------------------------------------------------------------------
struct AIConfig {
    std::optional<double> confidenceThreshold;
    std::optional<double> iouThreshold;
    std::optional<std::set<std::string>> classesEnabled;
    std::optional<bool> detectionEnabled;
    std::optional<bool> crowdCountingEnabled;
    std::optional<bool> showOverlay;

    std::optional<int> crowdInferInterval;
    std::optional<int> crowdInputWidth;
    std::optional<int> crowdInputHeight;
    std::optional<int> inferSize;
    std::optional<int> bitrateKbps;
    std::optional<int> reconnectIntervalMs;
};
// ---------------- Runtime AI config, diubah live via APIController ----------------
struct RuntimeAIConfig {
    std::mutex mtx;
    float confidenceThreshold;
    std::set<int> targetClasses;   // kosong = semua class diizinkan
};

// ------------------------------------------------------------------
// APIController
//
// Mengelola koneksi WebSocket ke GCS: broadcast data (hasil per-frame &
// konfigurasi) dan menangani command masuk (ubah konfigurasi AI).
//
// Format pesan broadcast (key JSON singkat untuk hemat bandwidth):
//
//   "result" -- dikirim tiap frame (broadcastResult):
//   {
//     "t": "result",
//     "ts": "2026-08-10T03:20:10.500Z",
//     "f": 10250,
//     "res": { "w": 1280, "h": 720 },
//     "inf": 25.1,
//     "fps": 24.7,
//     "det": [
//       { "id": 1, "cls": "person", "cf": 0.87, "bbox": [0.42, 0.31, 0.12, 0.25] }
//     ],
//     "crwd": { "cnt": 1000, "lvl": "crowded" }
//   }
//
//   "config" -- dikirim saat konfigurasi berubah / sinkronisasi status
//   (broadcastConfig):
//   {
//     "t": "config",
//     "ts": "2026-08-10T03:20:10.500Z",
//     "conf": 0.5,
//     "nms": 0.45,
//     "cls": ["person", "vehicle"],
//     "overlay": true,
//     "det_on": true,
//     "crwd_on": true
//   }
// ------------------------------------------------------------------
class APIController {
public:
    using ConfigCommandHandler = std::function<bool(const AIConfig&)>;

    APIController();

    void setConfigCommandHandler(ConfigCommandHandler handler);

    // Daftarkan Config master + path file-nya, supaya APIController bisa
    // auto-save ke config.ini setiap kali command berhasil diterapkan.
    // cfg harus tetap valid (hidup) selama APIController dipakai.
    void setConfigStore(Config* cfg, const std::string& configPath);

    void run(uint16_t port);

    // Broadcast pesan "result": hasil deteksi + crowd counting untuk satu
    // frame, termasuk fps saat ini. Dikirim tiap frame (real-time).
    void broadcastResult(long frame_id,
                          int res_w, int res_h,
                          const std::vector<ApiDetectionResult>& dets,
                          double inference_time_ms,
                          double fps,
                          const ApiCrowdResult& crowd);

    // Broadcast pesan "config": konfigurasi & status modul AI saat ini.
    // Dikirim saat konfigurasi berubah atau untuk sinkronisasi berkala,
    // bukan tiap frame.
    void broadcastConfig(double currentConfidenceThreshold,
                          double currentIouThreshold,
                          const std::set<std::string>& currentClasses,
                          bool showOverlay,
                          bool detection_enabled,
                          bool crowd_counting_enabled);

private:
    void onOpen(ConnHandle hdl);
    void onClose(ConnHandle hdl);
    void onMessage(ConnHandle hdl, WsServer::message_ptr msg);
    void handleConfigCommand(ConnHandle hdl, const json& j);
    void broadcastRaw(const std::string& payload);

    // Update field yang relevan di Config master berdasarkan AIConfig
    // yang baru diterapkan, lalu simpan ke file.
    void persistConfig(const AIConfig& newCfg);

    WsServer server_;
    std::set<ConnHandle, std::owner_less<ConnHandle>> clients_;
    std::mutex mutex_;
    ConfigCommandHandler onConfigCommand_;

    // Config store untuk auto-save (opsional; kalau tidak di-set, auto-save dilewati)
    Config* appConfig_ = nullptr;
    std::string configPath_;
    std::mutex configMutex_;   // proteksi terpisah untuk appConfig_ (bisa diakses thread lain juga)

    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);
};