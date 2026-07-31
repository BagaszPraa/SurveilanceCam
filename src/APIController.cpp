// APIController.cpp

#include "APIController.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
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
void APIController::broadcastDetections(long frame_id,
                                         int res_w, int res_h,
                                         const std::vector<ApiDetection>& dets,
                                         double inference_time_ms) {
    json j;
    j["type"] = "detection_result";
    j["timestamp"] = isoTimestampNow();
    j["frame_id"] = frame_id;
    j["resolution"] = {{"width", res_w}, {"height", res_h}};
    j["inference_time_ms"] = inference_time_ms;

    json arr = json::array();
    for (const auto& d : dets) {
        json jd;
        jd["id"] = d.id;
        jd["class"] = d.cls;
        jd["confidence"] = d.confidence;
        jd["bbox"] = {
            {"x", d.bbox_x}, {"y", d.bbox_y},
            {"w", d.bbox_w}, {"h", d.bbox_h}
        };
        jd["bbox_format"] = "xywh_normalized";
        arr.push_back(jd);
    }
    j["detections"] = arr;

    broadcastRaw(j.dump());
}

void APIController::broadcastStatus(const std::string& model_name,
                                     double fps,
                                     const AIConfig& cfg) {
    json j;
    j["type"] = "ai_status";
    j["timestamp"] = isoTimestampNow();
    j["status"] = "running";
    j["model"] = model_name;
    j["fps"] = fps;
    j["current_threshold"] = cfg.confidence_threshold;

    json classes = json::array();
    for (const auto& c : cfg.classes_enabled) classes.push_back(c);
    j["active_classes"] = classes;

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

        if (type == "config_command") {
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
// ------------------------------------------------------------------
void APIController::persistConfig(const AIConfig& newCfg) {
    if (!appConfig_) {
        logWarn("Config store belum di-set, auto-save dilewati.");
        return;
    }

    std::lock_guard<std::mutex> lock(configMutex_);

    appConfig_->confThresh = newCfg.confidence_threshold;
    appConfig_->nmsThresh  = newCfg.iou_threshold;

    // Konversi set<string> (class ID dalam bentuk string) -> vector<int>
    appConfig_->targetClasses.clear();
    for (const auto& s : newCfg.classes_enabled) {
        try {
            appConfig_->targetClasses.push_back(std::stoi(s));
        } catch (const std::exception&) {
            logWarn("classes_enabled berisi nilai non-numerik, diabaikan: " + s);
        }
    }

    if (ConfigManager::saveToFile(configPath_, *appConfig_)) {
        logInfo("Config berhasil di-auto-save ke: " + configPath_);
    } else {
        logError("Gagal auto-save config ke: " + configPath_);
    }
}
void APIController::handleConfigCommand(ConnHandle hdl, const json& j) {
    AIConfig cfg;
    if (!j.contains("params")) {
        logWarn("config_command tanpa field 'params', command_id=" + j.value("command_id", ""));
        // kirim ack rejected, return
    }
    const auto& params = j.at("params");
    logInfo("Menerima config_command id=" + j.value("command_id", "") +
             " params=" + params.dump());

    if (params.contains("confidence_threshold"))
        cfg.confidence_threshold = params["confidence_threshold"];
    if (params.contains("iou_threshold"))
        cfg.iou_threshold = params["iou_threshold"];
    if (params.contains("classes_enabled")) {
        cfg.classes_enabled.clear();
        for (const auto& c : params["classes_enabled"])
            cfg.classes_enabled.insert(c.get<std::string>());
    }

    bool applied = onConfigCommand_ ? onConfigCommand_(cfg) : false;

    if (applied) {
        logInfo("Config command diterapkan: " + j.value("command_id", ""));
        persistConfig(cfg);   // <-- auto-save di sini
    } else {
        logWarn("Config command ditolak: " + j.value("command_id", ""));
    }

    json ack;
    ack["type"] = "config_ack";
    ack["command_id"] = j.value("command_id", "");
    ack["status"] = applied ? "applied" : "rejected";
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