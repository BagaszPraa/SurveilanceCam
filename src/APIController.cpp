// APIController.cpp

#include "APIController.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <array>

// ---------------------------------------------------
// Logging
// ---------------------------------------------------
void APIController::logInfo(const std::string& msg)
{
    std::cout << "[APIController] [INFO] " << msg << std::endl;
}

void APIController::logWarn(const std::string& msg)
{
    std::cout << "[APIController] [WARN] " << msg << std::endl;
}

void APIController::logError(const std::string& msg)
{
    std::cerr << "[APIController] [ERROR] " << msg << std::endl;
}

// ------------------------------------------------------------------
// Helper: timestamp ISO8601 UTC
// ------------------------------------------------------------------
namespace {

std::string isoTimestampNow() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm_utc{};
    gmtime_r(&t, &tm_utc);

    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

// ------------------------------------------------------------------
// Whitelist key FULL-DEV yang SENGAJA ditolak kalau muncul di params
// "cmd". Parameter ini hanya boleh diubah lewat edit config.ini manual
// + restart, bukan lewat WebSocket -- alasan detail ada di komentar
// struct AIConfig (APIController.h).
// ------------------------------------------------------------------
const std::array<std::string, 3> kFullDevOnlyKeys = {
    "ip_address", "api_port", "api_host"
};

}  // namespace

// ------------------------------------------------------------------
// Constructor
// ------------------------------------------------------------------
APIController::APIController() {
    server_.init_asio();
    server_.set_reuse_addr(true);

    server_.set_open_handler(
        [this](ConnHandle hdl) { onOpen(hdl); });
    server_.set_close_handler(
        [this](ConnHandle hdl) { onClose(hdl); });
    server_.set_message_handler(
        [this](ConnHandle hdl, WsServer::message_ptr msg) {
            onMessage(hdl, msg);
        });

    // Kurangi log verbose bawaan websocketpp
    server_.clear_access_channels(websocketpp::log::alevel::all);
    server_.set_access_channels(websocketpp::log::alevel::connect |
                                 websocketpp::log::alevel::disconnect |
                                 websocketpp::log::alevel::app);
}
void APIController::setConfigStore(Config* cfg, const std::string& configPath) {
    appConfig_ = cfg;
    configPath_ = configPath;
}

void APIController::setConfigCommandHandler(ConfigCommandHandler handler) {
    onConfigCommand_ = std::move(handler);
}

void APIController::run(uint16_t port) {
    try {
        server_.listen(boost::asio::ip::tcp::v4(), port);
        server_.start_accept();
        logInfo("Berjalan di port " + std::to_string(port));
        server_.run();
    } catch (const std::exception& e) {
        logError("Gagal menjalankan server di port " + std::to_string(port) + ": " + e.what());
        throw; // atau exit, sesuai kebutuhan
    }
}

// ------------------------------------------------------------------
// Broadcast
// ------------------------------------------------------------------

// Pesan "result": hasil deteksi + crowd counting untuk satu frame,
// beserta fps saat ini. Key JSON sengaja dipersingkat untuk hemat
// bandwidth (dikirim tiap frame). Lihat komentar format di
// APIController.h.
void APIController::broadcastResult(long frame_id,
                                     int res_w, int res_h,
                                     const std::vector<ApiDetectionResult>& dets,
                                     double inference_time_ms,
                                     double fps,
                                     const ApiCrowdResult& crowd) {
    json j;
    j["t"] = "result";
    j["ts"] = isoTimestampNow();
    j["f"] = frame_id;
    j["res"] = {{"w", res_w}, {"h", res_h}};
    j["inf"] = inference_time_ms;
    j["fps"] = fps;

    json arr = json::array();
    for (const auto& d : dets) {
        json jd;
        jd["id"] = d.id;
        jd["cls"] = d.cls;
        jd["cf"] = d.confidence;
        jd["bbox"] = { d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h };
        arr.push_back(jd);
    }
    j["det"] = arr;

    j["crwd"] = {
        {"cnt", crowd.count},
        {"lvl", crowd.densityLevel}
    };

    broadcastRaw(j.dump());
}

// Pesan "config": konfigurasi & status modul AI saat ini. Dikirim saat
// konfigurasi berubah / sinkronisasi berkala, bukan tiap frame -- jadi
// tidak perlu sehemat "result".
void APIController::broadcastConfig(double currentConfidenceThreshold,
                                     double currentIouThreshold,
                                     const std::set<std::string>& currentClasses,
                                     bool showOverlay,
                                     bool detection_enabled,
                                     bool crowd_counting_enabled) {
    json j;
    j["t"] = "config";
    j["ts"] = isoTimestampNow();
    j["conf"] = currentConfidenceThreshold;
    j["nms"] = currentIouThreshold;

    json classes = json::array();
    for (const auto& c : currentClasses) classes.push_back(c);
    j["cls"] = classes;

    j["overlay"] = showOverlay;
    j["det_on"] = detection_enabled;
    j["crwd_on"] = crowd_counting_enabled;

    broadcastRaw(j.dump());
}

// ------------------------------------------------------------------
// Connection lifecycle
// ------------------------------------------------------------------
void APIController::onOpen(ConnHandle hdl) {
    std::lock_guard<std::mutex> lock(mutex_);
    clients_.insert(hdl);
    logInfo("Klien baru terhubung. Total klien: " + std::to_string(clients_.size()));
}
void APIController::onClose(ConnHandle hdl) {
    std::lock_guard<std::mutex> lock(mutex_);
    clients_.erase(hdl);
    logInfo("Klien terputus. Total klien: " + std::to_string(clients_.size()));
}
// ------------------------------------------------------------------
// Message handling
// ------------------------------------------------------------------
void APIController::onMessage(ConnHandle hdl, WsServer::message_ptr msg) {
    try {
        json j = json::parse(msg->get_payload());
        const std::string type = j.value("type", "");

        if (type == "cmd") {
            handleConfigCommand(hdl, j);
        }
        else {
            logWarn("Tipe pesan tidak dikenal: " + type);
        }
        // tipe pesan lain (mis. "ping") bisa ditambah di sini
    } catch (const std::exception& e) {
        server_.get_alog().write(websocketpp::log::alevel::app,
            std::string("Failed to parse message: ") + e.what());
    }
}
// ------------------------------------------------------------------
// Sinkronkan AIConfig (dari command GCS) ke Config master, lalu simpan
// ke file config.ini. Kalau appConfig_ belum di-set (setConfigStore
// belum dipanggil), auto-save dilewati.
//
// Semua field opsional -- hanya field yang benar-benar dikirim di
// command ini yang ditulis ke Config master, sisanya dipertahankan.
// ------------------------------------------------------------------
void APIController::persistConfig(const AIConfig& newCfg) {
    if (!appConfig_) {
        logWarn("Config store belum di-set, auto-save dilewati.");
        return;
    }

    std::lock_guard<std::mutex> lock(configMutex_);

    // ---- Golongan USER ----
    if (newCfg.confidenceThreshold.has_value()) {
        appConfig_->confThresh = newCfg.confidenceThreshold.value();
    }
    if (newCfg.iouThreshold.has_value()) {
        appConfig_->nmsThresh = newCfg.iouThreshold.value();
    }
    if (newCfg.classesEnabled.has_value()) {
        appConfig_->targetClasses.clear();
        for (const auto& s : newCfg.classesEnabled.value()) {
            try {
                appConfig_->targetClasses.push_back(std::stoi(s));
            } catch (const std::exception&) {
                logWarn("class berisi nilai non-numerik, diabaikan: " + s);
            }
        }
    }
    if (newCfg.detectionEnabled.has_value()) {
        appConfig_->isDetection = newCfg.detectionEnabled.value();
    }
    if (newCfg.crowdCountingEnabled.has_value()) {
        appConfig_->isCrowdCounting = newCfg.crowdCountingEnabled.value();
    }
    if (newCfg.showOverlay.has_value()) {
        appConfig_->showOverlay = newCfg.showOverlay.value();
    }

    // ---- Golongan SEMI-DEV ----
    if (newCfg.crowdInferInterval.has_value()) {
        appConfig_->crowdInferInterval = newCfg.crowdInferInterval.value();
    }
    if (newCfg.crowdInputWidth.has_value()) {
        appConfig_->crowdInputWidth = newCfg.crowdInputWidth.value();
    }
    if (newCfg.crowdInputHeight.has_value()) {
        appConfig_->crowdInputHeight = newCfg.crowdInputHeight.value();
    }
    if (newCfg.inferSize.has_value()) {
        appConfig_->inferSize = newCfg.inferSize.value();
    }
    if (newCfg.bitrateKbps.has_value()) {
        appConfig_->bitrateKbps = newCfg.bitrateKbps.value();
    }
    if (newCfg.reconnectIntervalMs.has_value()) {
        appConfig_->reconnectIntervalMs = newCfg.reconnectIntervalMs.value();
    }

    if (ConfigManager::saveToFile(configPath_, *appConfig_)) {
        logInfo("Config berhasil di-auto-save ke: " + configPath_);
    } else {
        logError("Gagal auto-save config ke: " + configPath_);
    }
}
void APIController::handleConfigCommand(ConnHandle hdl, const json& j) {
    const std::string commandId = j.value("command_id", "");

    if (!j.contains("params")) {
        logWarn("cmd tanpa field 'params', command_id=" + commandId);

        json ack;
        ack["type"] = "config_ack";
        ack["command_id"] = commandId;
        ack["status"] = "rejected";
        ack["reason"] = "missing 'params' field";
        ack["timestamp"] = isoTimestampNow();

        websocketpp::lib::error_code ec;
        server_.send(hdl, ack.dump(), websocketpp::frame::opcode::text, ec);
        if (ec) {
            logError("Gagal mengirim ack: " + ec.message());
        }
        return; // jangan lanjut ke j.at("params"), field-nya memang tidak ada
    }

    const auto& params = j.at("params");
    logInfo("Menerima cmd id=" + commandId +
             " params=" + params.dump());

    // ---- Golongan FULL-DEV: tolak eksplisit kalau ada yang mencoba ----
    // Parameter ini sengaja tidak boleh diubah lewat WebSocket sama
    // sekali (lihat penjelasan di AIConfig / APIController.h) --
    // mengubah alamat/port API dari koneksi API itu sendiri berisiko
    // self-lockout. Command langsung ditolak total, tidak parsial.
    std::vector<std::string> blockedKeys;
    for (const auto& key : kFullDevOnlyKeys) {
        if (params.contains(key)) blockedKeys.push_back(key);
    }
    if (!blockedKeys.empty()) {
        std::string joined;
        for (size_t i = 0; i < blockedKeys.size(); ++i) {
            joined += blockedKeys[i];
            if (i + 1 < blockedKeys.size()) joined += ", ";
        }
        logWarn("cmd id=" + commandId +
                " ditolak, berisi parameter full-dev-only: " + joined);

        json ack;
        ack["type"] = "config_ack";
        ack["command_id"] = commandId;
        ack["status"] = "rejected";
        ack["reason"] = "parameter berikut hanya bisa diubah lewat config.ini manual + restart: " + joined;
        ack["timestamp"] = isoTimestampNow();

        websocketpp::lib::error_code ec;
        server_.send(hdl, ack.dump(), websocketpp::frame::opcode::text, ec);
        if (ec) {
            logError("Gagal mengirim ack: " + ec.message());
        }
        return;
    }

    AIConfig cfg;
    bool requiresRestart = false;

    // ---- Golongan USER: real-time, tanpa restart ----
    if (params.contains("conf"))
        cfg.confidenceThreshold = params["conf"].get<double>();
    if (params.contains("nms"))
        cfg.iouThreshold = params["nms"].get<double>();
    if (params.contains("class")) {
        std::set<std::string> classes;
        for (const auto& c : params["class"])
            classes.insert(c.get<std::string>());
        cfg.classesEnabled = std::move(classes);
    }
    if (params.contains("detect"))
        cfg.detectionEnabled = params["detect"].get<bool>();
    if (params.contains("crowd"))
        cfg.crowdCountingEnabled = params["crowd"].get<bool>();
    if (params.contains("overlay"))
        cfg.showOverlay = params["overlay"].get<bool>();

    // ---- Golongan SEMI-DEV: ditulis ke config, sebagian butuh restart ----
    if (params.contains("crowd_interval")) {
        cfg.crowdInferInterval = params["crowd_interval"].get<int>();
        requiresRestart = true; // interval saat ini di-cache sebagai local const di main.cpp
    }
    if (params.contains("crowd_width")) {
        cfg.crowdInputWidth = params["crowd_width"].get<int>();
        requiresRestart = true; // buffer CrowdCounting dialokasikan sesuai ukuran ini saat startup
    }
    if (params.contains("crowd_height")) {
        cfg.crowdInputHeight = params["crowd_height"].get<int>();
        requiresRestart = true;
    }
    if (params.contains("infer_size")) {
        cfg.inferSize = params["infer_size"].get<int>();
        requiresRestart = true; // engine YOLO di-load dengan ukuran ini saat startup
    }
    if (params.contains("bitrate_kbps")) {
        cfg.bitrateKbps = params["bitrate_kbps"].get<int>();
        requiresRestart = true; // encoder RTSP dikonfigurasi saat construct RtspServer
    }
    if (params.contains("reconnect_ms")) {
        cfg.reconnectIntervalMs = params["reconnect_ms"].get<int>();
        // TIDAK butuh restart -- dibaca langsung dari Config master tiap
        // iterasi reconnect loop di main.cpp.
    }

    bool applied = onConfigCommand_ ? onConfigCommand_(cfg) : false;

    if (applied) {
        logInfo("Cmd diterapkan: " + commandId +
                (requiresRestart ? " (sebagian parameter butuh restart aplikasi untuk berlaku penuh)" : ""));
        persistConfig(cfg);   // <-- auto-save di sini
    } else {
        logWarn("Cmd ditolak: " + commandId);
    }

    json ack;
    ack["type"] = "config_ack";
    ack["command_id"] = commandId;
    ack["status"] = applied ? "applied" : "rejected";
    ack["requires_restart"] = applied && requiresRestart;
    ack["timestamp"] = isoTimestampNow();

    websocketpp::lib::error_code ec;
    server_.send(hdl, ack.dump(), websocketpp::frame::opcode::text, ec);
    if (ec) {
        logError("Gagal mengirim ack: " + ec.message());
    }
}

// ------------------------------------------------------------------
// Broadcast helper
// ------------------------------------------------------------------
void APIController::broadcastRaw(const std::string& payload) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& hdl : clients_) {
        websocketpp::lib::error_code ec;
        server_.send(hdl, payload, websocketpp::frame::opcode::text, ec);
        if (ec) {
            logError("Gagal broadcast ke salah satu klien: " + ec.message());
        }
    }
}