// ConfigManager.cpp

#include "ConfigManager.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>

// ---------------------------------------------------
// Logging
// ---------------------------------------------------
void ConfigManager::logInfo(const std::string& msg)
{
    std::cout << "[ConfigManager] [INFO] " << msg << std::endl;
}

void ConfigManager::logWarn(const std::string& msg)
{
    std::cout << "[ConfigManager] [WARN] " << msg << std::endl;
}

void ConfigManager::logError(const std::string& msg)
{
    std::cerr << "[ConfigManager] [ERROR] " << msg << std::endl;
}

// ------------------------------------------------------------------
// Helper parsing internal
// ------------------------------------------------------------------
std::string ConfigManager::trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::vector<int> ConfigManager::parseIntList(const std::string& value) {
    std::vector<int> result;
    std::stringstream ss(value);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) result.push_back(std::stoi(item));
    }
    return result;
}

bool ConfigManager::parseBool(const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "1" || v == "true" || v == "yes" || v == "on");
}

EncoderType ConfigManager::parseEncoderType(const std::string& value, EncoderType fallback) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "cpu")                      return EncoderType::CPU;
    if (v == "nvidia_gpu" || v == "gpu") return EncoderType::NVIDIA_GPU;
    if (v == "jetson")                   return EncoderType::JETSON;
    logWarn("encoder_type '" + value + "' tidak dikenal, memakai default.");
    return fallback;
}

CodecType ConfigManager::parseCodecType(const std::string& value, CodecType fallback) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    if (v == "h264") return CodecType::H264;
    if (v == "h265") return CodecType::H265;
    logWarn("codec_type '" + value + "' tidak dikenal, memakai default.");
    return fallback;
}
std::string ConfigManager::encoderTypeToString(EncoderType type) {
    switch (type) {
        case EncoderType::CPU:        return "cpu";
        case EncoderType::NVIDIA_GPU: return "nvidia_gpu";
        case EncoderType::JETSON:     return "jetson";
        default:                      return "cpu";
    }
}
std::string ConfigManager::codecTypeToString(CodecType type) {
    switch (type) {
        case CodecType::H264: return "h264";
        case CodecType::H265: return "h265";
        default:              return "h264";
    }
}

std::string ConfigManager::findConfigPath(int argc, char* argv[]) {
    std::string configPath = "../config.ini";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            configPath = argv[i + 1];
            break;
        }
    }
    return configPath;
}

// ------------------------------------------------------------------
// Baca file config bergaya INI sederhana
//   # komentar
//   key = value
// Baris kosong dan yang diawali '#' atau ';' diabaikan.
// ------------------------------------------------------------------
bool ConfigManager::loadFromFile(const std::string& path, Config& cfg) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;  // file tidak ada -> pakai default, bukan error fatal
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
            if (key == "input")                      cfg.inputSource = value;
            else if (key == "model")                 cfg.modelPath = value;
            else if (key == "width")                 cfg.width = std::stoi(value);
            else if (key == "height")                cfg.height = std::stoi(value);
            else if (key == "fps")                   cfg.fps = std::stoi(value);
            else if (key == "infer_size")             cfg.inferSize = std::stoi(value);
            else if (key == "conf")                  cfg.confThresh = std::stof(value);
            else if (key == "nms")                   cfg.nmsThresh = std::stof(value);
            else if (key == "classes_enabled")        cfg.targetClasses = parseIntList(value);
            else if (key == "rtsp_port")             cfg.rtspPort = std::stoi(value);
            else if (key == "rtsp_mount")            cfg.rtspMount = value;
            else if (key == "ip_address")            cfg.ipAdress = value;
            else if (key == "gstreamer")             cfg.isGstreamer = parseBool(value);
            else if (key == "reconnect_interval_ms") cfg.reconnectIntervalMs = std::stoi(value);
            else if (key == "show_overlay")          cfg.showOverlay = parseBool(value);
            else if (key == "encoder_type")          cfg.encoderType = parseEncoderType(value, cfg.encoderType);
            else if (key == "codec_type")            cfg.codecType = parseCodecType(value, cfg.codecType);
            else if (key == "bitrate_kbps")          cfg.bitrateKbps = std::stoi(value);
            else if (key == "api_port")              cfg.apiPort = std::stoi(value);
            else if (key == "api_host")              cfg.apiHost = value;
            else logWarn("Key tidak dikenal di baris " + std::to_string(lineNo) + ": " + key);
        } catch (const std::exception& e) {
            logWarn("Gagal parsing baris " + std::to_string(lineNo) + " (" + key + "=" + value + "): " + e.what());
        }
    }
    return true;
}

// ------------------------------------------------------------------
// Tulis konfigurasi saat ini ke file bergaya INI sederhana.
// Format & urutan key mengikuti loadFromFile agar file bisa dibaca ulang
// tanpa masalah.
// ------------------------------------------------------------------
bool ConfigManager::saveToFile(const std::string& path, const Config& cfg) {
    std::ofstream file(path);
    if (!file.is_open()) {
        logError("Gagal membuka file untuk menulis: " + path);
        return false;
    }

    file << "# Config ini digenerate otomatis oleh ConfigManager::saveToFile\n";
    file << "# Bisa juga diedit manual, format: key = value\n\n";

    file << "input = "  << cfg.inputSource << "\n";
    file << "model = "  << cfg.modelPath   << "\n";
    file << "width = "  << cfg.width       << "\n";
    file << "height = " << cfg.height      << "\n";
    file << "fps = "    << cfg.fps         << "\n";
    file << "infer_size = " << cfg.inferSize << "\n";
    file << "conf = " << cfg.confThresh << "\n";
    file << "nms = "  << cfg.nmsThresh  << "\n";

    if (!cfg.targetClasses.empty()) {
        file << "classes_enabled = ";
        for (size_t i = 0; i < cfg.targetClasses.size(); ++i) {
            file << cfg.targetClasses[i];
            if (i + 1 < cfg.targetClasses.size()) file << ",";
        }
        file << "\n";
    }

    file << "rtsp_port = "  << cfg.rtspPort  << "\n";
    file << "rtsp_mount = " << cfg.rtspMount << "\n";
    file << "ip_address = " << cfg.ipAdress  << "\n";
    file << "gstreamer = "  << (cfg.isGstreamer ? "true" : "false") << "\n";
    file << "reconnect_interval_ms = " << cfg.reconnectIntervalMs << "\n";
    file << "show_overlay = " << (cfg.showOverlay ? "true" : "false") << "\n";
    file << "encoder_type = " << encoderTypeToString(cfg.encoderType) << "\n";
    file << "codec_type = "   << codecTypeToString(cfg.codecType)     << "\n";
    file << "bitrate_kbps = " << cfg.bitrateKbps << "\n";
    file << "api_port = " << cfg.apiPort << "\n";
    file << "api_host = " << cfg.apiHost << "\n";

    file.close();

    if (file.fail()) {
        logError("Terjadi kesalahan saat menulis konfigurasi ke: " + path);
        return false;
    }

    logInfo("Konfigurasi berhasil disimpan ke: " + path);
    return true;
}

void ConfigManager::applyCliArgs(int argc, char* argv[], Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto nextVal = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string(def);
        };

        if (arg == "--config")               nextVal("");  // sudah ditangani di findConfigPath()
        else if (arg == "--input")           cfg.inputSource = nextVal(cfg.inputSource.c_str());
        else if (arg == "--model")           cfg.modelPath = nextVal(cfg.modelPath.c_str());
        else if (arg == "--width")           cfg.width = std::stoi(nextVal(std::to_string(cfg.width).c_str()));
        else if (arg == "--height")          cfg.height = std::stoi(nextVal(std::to_string(cfg.height).c_str()));
        else if (arg == "--fps")             cfg.fps = std::stoi(nextVal(std::to_string(cfg.fps).c_str()));
        else if (arg == "--infer-size")      cfg.inferSize = std::stoi(nextVal(std::to_string(cfg.inferSize).c_str()));
        else if (arg == "--conf")            cfg.confThresh = std::stof(nextVal(std::to_string(cfg.confThresh).c_str()));
        else if (arg == "--nms")             cfg.nmsThresh = std::stof(nextVal(std::to_string(cfg.nmsThresh).c_str()));
        else if (arg == "--rtsp-port")       cfg.rtspPort = std::stoi(nextVal(std::to_string(cfg.rtspPort).c_str()));
        else if (arg == "--rtsp-mount")      cfg.rtspMount = nextVal(cfg.rtspMount.c_str());
        else if (arg == "--gstreamer")       cfg.isGstreamer = true;
        else if (arg == "--no-gstreamer")    cfg.isGstreamer = false;
        else if (arg == "--reconnect-interval") cfg.reconnectIntervalMs = std::stoi(nextVal(std::to_string(cfg.reconnectIntervalMs).c_str()));
        else if (arg == "--overlay")         cfg.showOverlay = true;
        else if (arg == "--no-overlay")      cfg.showOverlay = false;
        else if (arg == "--encoder")         cfg.encoderType = parseEncoderType(nextVal(""), cfg.encoderType);
        else if (arg == "--codec")           cfg.codecType = parseCodecType(nextVal(""), cfg.codecType);
        else if (arg == "--bitrate")         cfg.bitrateKbps = std::stoi(nextVal(std::to_string(cfg.bitrateKbps).c_str()));
        else if (arg == "--api-port")        cfg.apiPort = std::stoi(nextVal(std::to_string(cfg.apiPort).c_str()));
        else if (arg == "--api-host")        cfg.apiHost = nextVal(cfg.apiHost.c_str());
    }
}

void ConfigManager::logSummary(const Config& cfg) {
    logInfo("==========================================");
    logInfo("Konfigurasi aktif:");
    logInfo("Input      : " + cfg.inputSource + " (gstreamer=" + (cfg.isGstreamer ? "true" : "false") + ")");
    logInfo("IP Address : " + cfg.ipAdress);
    logInfo("Model      : " + cfg.modelPath);
    logInfo("Resolusi   : " + std::to_string(cfg.width) + "x" + std::to_string(cfg.height) + " @ " +
            std::to_string(cfg.fps) + "fps (permintaan awal, resolusi final mengikuti source)");
    logInfo("InferSize  : " + std::to_string(cfg.inferSize));
    logInfo("Conf/NMS   : " + std::to_string(cfg.confThresh) + " / " + std::to_string(cfg.nmsThresh));

    if (cfg.targetClasses.empty()) {
        logInfo("Classes    : semua class aktif");
    } else {
        std::string classesStr;
        for (size_t i = 0; i < cfg.targetClasses.size(); ++i) {
            classesStr += std::to_string(cfg.targetClasses[i]);
            if (i + 1 < cfg.targetClasses.size()) classesStr += ",";
        }
        logInfo("Classes    : " + classesStr + " (" + std::to_string(cfg.targetClasses.size()) + " class aktif)");
    }

    logInfo("RTSP       : port " + std::to_string(cfg.rtspPort) + ", mount " + cfg.rtspMount);
    logInfo("Encoder    : " + encoderTypeToString(cfg.encoderType) +
            ", codec " + codecTypeToString(cfg.codecType) +
            ", bitrate " + std::to_string(cfg.bitrateKbps) + " kbps");
    logInfo("API        : " + cfg.apiHost + ":" + std::to_string(cfg.apiPort));
    logInfo("Reconnect  : tiap " + std::to_string(cfg.reconnectIntervalMs) + " ms jika input belum/tidak tersedia");
    logInfo(std::string("Overlay    : ") + (cfg.showOverlay ? "aktif" : "nonaktif") +
            " (teks info Count/Infer/FPS/Resolusi; bbox tetap selalu digambar)");
    logInfo("==========================================");
}

// ------------------------------------------------------------------
// Alur lengkap
// ------------------------------------------------------------------

Config ConfigManager::load(int argc, char* argv[], std::string& outConfigPath) {
    Config cfg;

    std::string configPath = findConfigPath(argc, argv);
    outConfigPath = configPath;   // <-- baris baru, simpan path ke caller

    if (loadFromFile(configPath, cfg)) {
        logInfo("Berhasil memuat konfigurasi dari: " + configPath);
    } else {
        logInfo("File config '" + configPath +
                "' tidak ditemukan, memakai nilai default (bisa juga diatur lewat argumen CLI).");
    }

    applyCliArgs(argc, argv, cfg);
    logSummary(cfg);

    return cfg;
}