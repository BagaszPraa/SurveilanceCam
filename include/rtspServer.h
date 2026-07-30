#pragma once

#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <condition_variable>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

// ---------------------------------------------------
// Pilihan encoder hardware/software
// ---------------------------------------------------
enum class EncoderType
{
    CPU,          // x264enc / x265enc (software, jalan di CPU)
    NVIDIA_GPU,   // nvh264enc / nvh265enc (NVENC, laptop/desktop dGPU NVIDIA)
    JETSON        // nvv4l2h264enc / nvv4l2h265enc (NVENC on-chip, Jetson)
};

// ---------------------------------------------------
// Pilihan codec video
// ---------------------------------------------------
enum class CodecType
{
    H264,
    H265
};

class RtspServer
{
public:

    RtspServer(
        int port,
        const std::string& mountPoint,
        int width,
        int height,
        int fps,
        const std::string& host,
        EncoderType encoderType,
        CodecType codecType,
        int bitrateKbps
    );

    ~RtspServer();

    bool start();
    void stop();
    bool getLatestFrame(cv::Mat& frame);
    void pushFrame(const cv::Mat& frame);

    std::string url() const;

    // ---- Setter opsional, HARUS dipanggil sebelum start() ----
    // Kalau dipanggil setelah server berjalan, hanya akan mencatat warning
    // dan tidak mengubah apa pun (pipeline yang sudah jalan tidak bisa
    // diganti on-the-fly).
    void setEncoderType(EncoderType type);
    void setCodecType(CodecType type);
    void setBitrateKbps(int bitrateKbps);

private:

    bool createServer();
    void destroyServer();

    std::string buildPipeline();
    std::string encoderElementName() const;
    static bool elementAvailable(const std::string& name);

    static std::string encoderTypeToString(EncoderType type);
    static std::string codecTypeToString(CodecType type);

    // ---- Logging helper, prefix [rtspServer] ----
    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);

    static void mediaConfigure(
        GstRTSPMediaFactory *factory,
        GstRTSPMedia *media,
        gpointer user_data);

    static void needData(
        GstElement *appsrc,
        guint length,
        gpointer user_data);

private:
    std::string m_host;
    GstBuffer* matToBuffer(const cv::Mat& frame);
    std::atomic<bool> m_newFrame{false};
    std::thread m_streamThread;
    std::mutex m_appsrcMutex;
    std::condition_variable m_appsrcCv;

    bool m_appsrcReady = false;

    int m_port;
    int m_width;
    int m_height;
    int m_fps;

    std::string m_mountPoint;

    EncoderType m_encoderType;
    CodecType   m_codecType;
    int         m_bitrateKbps;

    GstRTSPServer* m_server = nullptr;
    GstRTSPMountPoints* m_mounts = nullptr;
    GstRTSPMediaFactory* m_factory = nullptr;

    GstElement* m_appsrc = nullptr;

    GMainLoop* m_loop = nullptr;

    std::thread m_mainLoopThread;

    std::mutex m_frameMutex;
    cv::Mat m_latestFrame;
    cv::Mat acquireLatestFrame();
    std::atomic<bool> m_running{false};

    guint m_sourceId = 0;

    uint64_t m_timestamp = 0;
};