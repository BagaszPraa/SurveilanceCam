#pragma once

#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/pbutils/pbutils.h>   // GstDiscoverer, dipakai buat probe codec H264/H265
#include <condition_variable>
#include "ConfigManager.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <mutex>

// Codec yang berhasil dideteksi dari stream RTSP
enum class VideoCodecType {
    H264,
    H265,
    UNKNOWN
};

// Backend decoder yang benar-benar tersedia di sistem (dicek lewat GStreamer registry,
// bukan cuma tebak-tebak dari nama platform)
enum class GstDecoderBackend {
    CPU,          // avdec_h264 / avdec_h265 (software, libav)
    NVIDIA_DGPU,  // nvh264dec / nvh265dec (plugin nvcodec, GPU discrete)
    JETSON,       // nvv4l2decoder (L4T multimedia API)
    UNKNOWN
};

class SourceHandler
{
public:
    bool openCapture(const Config& cfg, cv::VideoCapture& cap);

private:
    bool isNetworkStream(const std::string& src);

    // Bangun pipeline GStreamer utk RTSP: probe codec -> pilih decoder -> susun chain -> resize
    std::string buildRtspGstPipeline(const Config& cfg);

    // ---- Auto-detection ----
    VideoCodecType   probeRtspCodec(const std::string& url, int timeoutSec = 5);
    GstDecoderBackend detectDecoderBackend(VideoCodecType codec);
    bool              gstElementExists(const std::string& factoryName);

    // ---- Helper ----
    std::string codecToString(VideoCodecType c);
    std::string backendToString(GstDecoderBackend b);
    std::string joinPipeline(const std::vector<std::string>& elements);

    // ---- Logging helper, prefix [SourceHandler] ----
    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);

    // ---- Cache hasil probe codec per URL ----
    // GstDiscoverer di probeRtspCodec() bikin koneksi RTSP-nya SENDIRI
    // (terpisah dari koneksi capture via cap.open()). Tanpa cache, tiap kali
    // openCapture() dipanggil -- termasuk tiap percobaan reconnect -- RTSP
    // server di-connect DUA KALI (sekali oleh discoverer, sekali oleh
    // cap.open()) dan NVDEC/decoder session sempat di-init dua kali.
    // Cache ini membuat probing hanya terjadi sekali per URL; percobaan
    // reconnect berikutnya langsung pakai hasil yang sudah diketahui.
    std::map<std::string, VideoCodecType> m_codecCache;
    std::mutex m_codecCacheMutex;
};