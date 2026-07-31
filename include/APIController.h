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
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;
using WsServer = websocketpp::server<websocketpp::config::asio>;
using ConnHandle = websocketpp::connection_hdl;

// ------------------------------------------------------------------
// Struct hasil deteksi tunggal
// ------------------------------------------------------------------
struct ApiDetection {
    std::string id;
    std::string cls;         // "person", "vehicle", dll
    double confidence;
    double bbox_x, bbox_y, bbox_w, bbox_h;  // format: xywh, normalized 0..1
};

// ------------------------------------------------------------------
// Struct konfigurasi AI yang bisa diubah lewat command dari GCS
// ------------------------------------------------------------------
struct AIConfig {
    double confidence_threshold = 0.75;
    double iou_threshold = 0.45;
    std::set<std::string> classes_enabled = {"person", "vehicle"};
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
// Mengelola koneksi WebSocket ke GCS: broadcast data (deteksi & status)
// dan menangani command masuk (ubah konfigurasi AI).
// ------------------------------------------------------------------
class APIController {
public:
    using ConfigCommandHandler = std::function<bool(const AIConfig&)>;

    APIController();

    void setConfigCommandHandler(ConfigCommandHandler handler);

    // Daftarkan Config master + path file-nya, supaya APIController bisa
    // auto-save ke config.ini setiap kali config_command berhasil diterapkan.
    // cfg harus tetap valid (hidup) selama APIController dipakai.
    void setConfigStore(Config* cfg, const std::string& configPath);

    void run(uint16_t port);

    void broadcastDetections(long frame_id,
                              int res_w, int res_h,
                              const std::vector<ApiDetection>& dets,
                              double inference_time_ms);

    void broadcastStatus(const std::string& model_name,
                          double fps,
                          const AIConfig& cfg);

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