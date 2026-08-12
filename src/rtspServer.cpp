#include "rtspServer.h"

#include <iostream>

RtspServer::RtspServer(
    int port,
    const std::string& mountPoint,
    int width,
    int height,
    int fps,
    const std::string& host,
    EncoderType encoderType,
    CodecType codecType,
    int bitrateKbps)
    :
    m_host(host),
    m_port(port),
    m_width(width),
    m_height(height),
    m_fps(fps),
    m_mountPoint(mountPoint),
    m_encoderType(encoderType),
    m_codecType(codecType),
    m_bitrateKbps(bitrateKbps)
{
}

// Overload praktis: encoder = AUTO, codec default H264.
RtspServer::RtspServer(
    int port,
    const std::string& mountPoint,
    int width,
    int height,
    int fps,
    const std::string& host,
    int bitrateKbps)
    :
    RtspServer(
        port,
        mountPoint,
        width,
        height,
        fps,
        host,
        EncoderType::AUTO,
        CodecType::H264,
        bitrateKbps)
{
}

RtspServer::~RtspServer()
{
    stop();
}

// ---------------------------------------------------
// Logging
// ---------------------------------------------------
void RtspServer::logInfo(const std::string& msg)
{
    std::cout << "[rtspServer] [INFO] " << msg << std::endl;
}

void RtspServer::logWarn(const std::string& msg)
{
    std::cout << "[rtspServer] [WARN] " << msg << std::endl;
}

void RtspServer::logError(const std::string& msg)
{
    std::cerr << "[rtspServer] [ERROR] " << msg << std::endl;
}

std::string RtspServer::encoderTypeToString(EncoderType type)
{
    switch (type)
    {
        case EncoderType::AUTO:       return "AUTO";
        case EncoderType::CPU:        return "CPU";
        case EncoderType::NVIDIA_GPU: return "NVIDIA_GPU";
        case EncoderType::JETSON:     return "JETSON";
    }
    return "UNKNOWN";
}

std::string RtspServer::codecTypeToString(CodecType type)
{
    switch (type)
    {
        case CodecType::H264: return "H264";
        case CodecType::H265: return "H265";
    }
    return "UNKNOWN";
}

void RtspServer::setEncoderType(EncoderType type)
{
    if (m_running)
    {
        logWarn("setEncoderType() diabaikan karena server sudah berjalan. "
                "Panggil sebelum start().");
        return;
    }
    m_encoderType = type;
}

void RtspServer::setCodecType(CodecType type)
{
    if (m_running)
    {
        logWarn("setCodecType() diabaikan karena server sudah berjalan. "
                "Panggil sebelum start().");
        return;
    }
    m_codecType = type;
}

void RtspServer::setBitrateKbps(int bitrateKbps)
{
    if (m_running)
    {
        logWarn("setBitrateKbps() diabaikan karena server sudah berjalan. "
                "Panggil sebelum start().");
        return;
    }
    m_bitrateKbps = bitrateKbps;
}

bool RtspServer::start()
{
    if(m_running)
        return true;

    gst_init(nullptr,nullptr);

    // ---- Resolusi EncoderType::AUTO ke tipe konkret sebelum apa pun lain ----
    // Dilakukan di sini (bukan cuma di createServer()) supaya log info di bawah
    // dan seluruh alur berikutnya sudah melihat encoder yang benar-benar dipakai.
    if (m_encoderType == EncoderType::AUTO)
    {
        EncoderType detected = detectBestEncoder(m_codecType);
        logInfo("EncoderType::AUTO -> encoder terdeteksi & dipilih otomatis: "
                + encoderTypeToString(detected));
        m_encoderType = detected;
    }

    logInfo("Memulai RTSP server dengan encoder="
            + encoderTypeToString(m_encoderType)
            + ", codec=" + codecTypeToString(m_codecType)
            + ", bitrate=" + std::to_string(m_bitrateKbps) + " kbps");

    if(!createServer())
        return false;

    m_running = true;

    m_mainLoopThread = std::thread([this]()
    {
        logInfo("RTSP main-loop thread started");

        while (m_running)
        {
            g_main_context_iteration(nullptr, FALSE);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        logInfo("RTSP main-loop thread stopped");
    });

    m_streamThread = std::thread([this]()
    {
        logInfo("Streaming thread started");

        while (m_running)
        {
            cv::Mat img = acquireLatestFrame();

            if (img.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            //---------------------------------------
            // cek apakah client RTSP sudah connect
            //---------------------------------------

            GstElement* appsrc = nullptr;

            {
                std::lock_guard<std::mutex> lock(m_appsrcMutex);
                appsrc = m_appsrc;
            }

            if (!appsrc)
            {
                // belum ada client RTSP
                continue;
            }

            GstBuffer* buffer = matToBuffer(img);

            GstFlowReturn ret =
                gst_app_src_push_buffer(
                    GST_APP_SRC(appsrc),
                    buffer);

            if(ret != GST_FLOW_OK)
            {
                logWarn("Push buffer gagal, kode: " + std::to_string(static_cast<int>(ret)));
            }
        }

        logInfo("Streaming thread stopped");
    });

    logInfo("RTSP Server Started");
    logInfo("URL : " + url());

    return true;
}

void RtspServer::stop()
{
    if(!m_running)
        return;

    m_running = false;

    if(m_loop)
        g_main_loop_quit(m_loop);

    if(m_mainLoopThread.joinable())
        m_mainLoopThread.join();

    if (m_streamThread.joinable())
        m_streamThread.join();

    destroyServer();

    logInfo("RTSP Server Stopped");
}

GstBuffer* RtspServer::matToBuffer(const cv::Mat& frame)
{
    const size_t size = frame.total() * frame.elemSize();

    GstBuffer* buffer =
        gst_buffer_new_allocate(nullptr, size, nullptr);

    GstMapInfo map;

    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame.data, size);
    gst_buffer_unmap(buffer, &map);

    // Jangan set PTS/DTS/DURATION sendiri.

    return buffer;
}

cv::Mat RtspServer::acquireLatestFrame()
{
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if (!m_newFrame)
        return cv::Mat();

    m_newFrame = false;

    return m_latestFrame.clone();
}

void RtspServer::pushFrame(const cv::Mat& frame)
{
    if(frame.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(m_frameMutex);

        frame.copyTo(m_latestFrame);

        m_newFrame = true;
    }
}

bool RtspServer::getLatestFrame(cv::Mat& frame)
{
    std::lock_guard<std::mutex> lock(m_frameMutex);

    if(!m_newFrame)
        return false;

    m_latestFrame.copyTo(frame);

    m_newFrame = false;

    return true;
}

std::string RtspServer::url() const
{
    return "rtsp://"
            + m_host
            + ":"
            + std::to_string(m_port)
            + m_mountPoint;
}

// ---------------------------------------------------
// Cek apakah plugin/elemen GStreamer tersedia di sistem
// (mis. nvh265enc butuh plugin nvcodec ter-install).
// ---------------------------------------------------
bool RtspServer::elementAvailable(const std::string& name)
{
    if (!gst_is_initialized())
    {
        gst_init(nullptr, nullptr);
    }

    GstElementFactory* factory = gst_element_factory_find(name.c_str());
    if (factory)
    {
        gst_object_unref(factory);
        return true;
    }
    return false;
}

// ---------------------------------------------------
// Auto-detect encoder terbaik yang BENAR-BENAR tersedia di sistem
// (dicek lewat GStreamer registry, bukan tebak-tebak platform).
// Prioritas: JETSON (NVENC on-chip) > NVIDIA_GPU (NVENC dGPU) > CPU (software).
// ---------------------------------------------------
EncoderType RtspServer::detectBestEncoder(CodecType codec)
{
    // Jetson: nvv4l2h264enc / nvv4l2h265enc (L4T multimedia API)
    const std::string jetsonElem =
        (codec == CodecType::H264) ? "nvv4l2h264enc" : "nvv4l2h265enc";
    if (elementAvailable(jetsonElem))
    {
        return EncoderType::JETSON;
    }

    // NVIDIA dGPU: nvh264enc / nvh265enc (plugin nvcodec)
    const std::string nvElem =
        (codec == CodecType::H264) ? "nvh264enc" : "nvh265enc";
    if (elementAvailable(nvElem))
    {
        return EncoderType::NVIDIA_GPU;
    }

    // Fallback software: x264enc / x265enc
    // (tidak dicek elementAvailable() di sini -- kalau ini pun tidak ada,
    // createServer() punya safety-net error yang jelas soal plugin yang kurang)
    return EncoderType::CPU;
}

// ---------------------------------------------------
// Nama elemen encoder GStreamer sesuai kombinasi
// encoder type + codec type yang dipilih.
// ---------------------------------------------------
std::string RtspServer::encoderElementName() const
{
    switch (m_encoderType)
    {
        case EncoderType::CPU:
            return (m_codecType == CodecType::H264) ? "x264enc" : "x265enc";

        case EncoderType::NVIDIA_GPU:
            return (m_codecType == CodecType::H264) ? "nvh264enc" : "nvh265enc";

        case EncoderType::JETSON:
            return (m_codecType == CodecType::H264) ? "nvv4l2h264enc" : "nvv4l2h265enc";

        case EncoderType::AUTO:
            // Seharusnya sudah di-resolve ke tipe konkret di start(), sebelum
            // pernah sampai ke sini. Kalau somehow masih AUTO, aman-kan ke CPU.
            return (m_codecType == CodecType::H264) ? "x264enc" : "x265enc";
    }
    return "";
}

// ---------------------------------------------------
// Bangun pipeline GStreamer sesuai encoder & codec yang dipilih.
// ---------------------------------------------------
std::string RtspServer::buildPipeline()
{
    // Safety-net: kalau buildPipeline() ternyata dipanggil sebelum AUTO
    // di-resolve (mis. dipanggil manual di luar start()), resolve di sini.
    if (m_encoderType == EncoderType::AUTO)
    {
        logWarn("buildPipeline() dipanggil dengan EncoderType::AUTO yang belum "
                "di-resolve, mendeteksi otomatis sekarang.");
        m_encoderType = detectBestEncoder(m_codecType);
    }

    // ---- bagian parse & payloader menyesuaikan codec ----
    std::string codecParse;
    std::string codecPay;

    switch (m_codecType)
    {
        case CodecType::H264:
            codecParse = "h264parse";
            codecPay   = "rtph264pay";
            break;

        case CodecType::H265:
        default:
            codecParse = "h265parse";
            codecPay   = "rtph265pay";
            break;
    }

    // ---- bagian convert & encoder menyesuaikan encoder type ----
    std::string preConvert;
    std::string encoderElement;

    switch (m_encoderType)
    {
        case EncoderType::AUTO:
            // Tidak pernah tercapai (sudah di-resolve di atas), dibiarkan
            // kosong di sini hanya supaya switch tetap exhaustive.
            break;

        case EncoderType::CPU:
        {
            preConvert = "! videoconvert ! video/x-raw,format=I420 ";

            if (m_codecType == CodecType::H264)
            {
                encoderElement =
                    "! x264enc "
                    "tune=zerolatency "
                    "speed-preset=ultrafast "
                    "bitrate=" + std::to_string(m_bitrateKbps) + " "
                    "key-int-max=30 "
                    "bframes=0 "
                    "rc-lookahead=0 "
                    "sync-lookahead=0 "
                    "threads=1 ";
            }
            else
            {
                encoderElement =
                    "! x265enc "
                    "tune=zerolatency "
                    "speed-preset=ultrafast "
                    "bitrate=" + std::to_string(m_bitrateKbps) + " "
                    "key-int-max=30 "
                    "option-string=bframes=0:rc-lookahead=0:sync-lookahead=0:threads=1 ";
            }
            break;
        }

        case EncoderType::NVIDIA_GPU:
        {
            // NVENC via plugin nvcodec (laptop/desktop dGPU NVIDIA)
            preConvert = "! videoconvert ! video/x-raw,format=NV12 ";

            if (m_codecType == CodecType::H264)
            {
                encoderElement =
                    "! nvh264enc "
                    "preset=low-latency-hp "
                    "rc-mode=cbr "
                    "bitrate=" + std::to_string(m_bitrateKbps) + " "
                    "gop-size=30 "
                    "zerolatency=true ";
            }
            else
            {
                encoderElement =
                    "! nvh265enc "
                    "preset=low-latency-hp "
                    "rc-mode=cbr "
                    "bitrate=" + std::to_string(m_bitrateKbps) + " "
                    "gop-size=30 "
                    "zerolatency=true ";
            }
            break;
        }

        case EncoderType::JETSON:
        {
            // NVENC on-chip Jetson (elemen nvv4l2h26xenc) butuh input
            // di NVMM memory, jadi harus lewat nvvidconv setelah
            // videoconvert biasa.
            preConvert =
                "! videoconvert ! video/x-raw,format=I420 "
                "! nvvidconv "
                "! video/x-raw(memory:NVMM),format=I420 ";

            // CATATAN: properti "bitrate" di nvv4l2h26xenc satuannya
            // bps (bit per detik), BEDA dengan x264enc/nvh26xenc yang
            // pakai kbps. Makanya dikonversi dulu di sini.
            const long bitrateBps =
                static_cast<long>(m_bitrateKbps) * 1000;

            if (m_codecType == CodecType::H264)
            {
                encoderElement =
                    "! nvv4l2h264enc "
                    "control-rate=1 "
                    "bitrate=" + std::to_string(bitrateBps) + " "
                    "iframeinterval=30 "
                    "preset-level=1 "
                    "maxperf-enable=true "
                    "insert-sps-pps=true ";
            }
            else
            {
                encoderElement =
                    "! nvv4l2h265enc "
                    "control-rate=1 "
                    "bitrate=" + std::to_string(bitrateBps) + " "
                    "iframeinterval=30 "
                    "preset-level=1 "
                    "maxperf-enable=true "
                    "insert-sps-pps=true ";
            }
            break;
        }
    }

    std::string pipeline =
        "( "
        "appsrc name=mysrc "
        "is-live=true "
        "block=false "
        "format=time "
        "do-timestamp=true "

        "! queue "
        "max-size-buffers=1 "
        "max-size-bytes=0 "
        "max-size-time=0 "
        "leaky=downstream "

        + preConvert +
        encoderElement +

        "! " + codecParse + " "
        "config-interval=1 "

        "! " + codecPay + " "
        "config-interval=1 "
        "name=pay0 "
        "pt=96 "

        ")";

    return pipeline;
}

bool RtspServer::createServer()
{
    m_loop = g_main_loop_new(nullptr,FALSE);
    if(!m_loop)
    {
        logError("Cannot create GMainLoop");
        return false;
    }

    m_server = gst_rtsp_server_new();
    if(!m_server)
    {
        logError("Cannot create RTSP server");
        return false;
    }

    gst_rtsp_server_set_service(
        m_server,
        std::to_string(m_port).c_str());

    m_mounts =
        gst_rtsp_server_get_mount_points(
            m_server);

    m_factory =
        gst_rtsp_media_factory_new();

    gst_rtsp_media_factory_set_shared(
        m_factory,
        TRUE);

    // ---- Validasi ketersediaan elemen encoder, fallback ke CPU kalau tidak ada ----
    // (m_encoderType harusnya sudah konkret di sini karena AUTO di-resolve di
    // start(); ini tetap dipertahankan sebagai safety-net terakhir.)
    std::string wantedElement = encoderElementName();

    if (!elementAvailable(wantedElement))
    {
        logWarn("Elemen encoder '" + wantedElement + "' tidak ditemukan di sistem "
                "(encoder=" + encoderTypeToString(m_encoderType) +
                ", codec=" + codecTypeToString(m_codecType) +
                "). Mencoba fallback ke CPU encoder.");

        if (m_encoderType != EncoderType::CPU)
        {
            m_encoderType = EncoderType::CPU;
            wantedElement = encoderElementName();
        }

        if (!elementAvailable(wantedElement))
        {
            logError("Elemen encoder fallback '" + wantedElement +
                      "' juga tidak ditemukan. Pastikan plugin GStreamer terkait "
                      "(x264/x265, nvcodec, atau nvv4l2) sudah ter-install. "
                      "RTSP server tidak bisa dibuat.");
            return false;
        }

        logInfo("Fallback berhasil, memakai encoder: " + wantedElement);
    }

    std::string pipeline = buildPipeline();

    logInfo("Pipeline aktif: " + pipeline);

    gst_rtsp_media_factory_set_launch(
        m_factory,
        pipeline.c_str());

    g_signal_connect(
        m_factory,
        "media-configure",
        G_CALLBACK(RtspServer::mediaConfigure),
        this);

    gst_rtsp_mount_points_add_factory(
        m_mounts,
        m_mountPoint.c_str(),
        m_factory);

    m_sourceId =
        gst_rtsp_server_attach(
            m_server,
            nullptr);

    if(m_sourceId==0)
    {
        logError("Cannot attach RTSP server");
        return false;
    }

    return true;
}

void RtspServer::destroyServer()
{
    if(m_mounts)
    {
        g_object_unref(m_mounts);
        m_mounts=nullptr;
    }

    if(m_server)
    {
        g_object_unref(m_server);
        m_server=nullptr;
    }

    if(m_loop)
    {
        g_main_loop_unref(m_loop);
        m_loop=nullptr;
    }

    m_factory=nullptr;
    m_appsrc=nullptr;
}

void RtspServer::mediaConfigure(
    GstRTSPMediaFactory *,
    GstRTSPMedia *media,
    gpointer user_data)
{
    RtspServer *self =
        static_cast<RtspServer *>(user_data);

    GstElement *pipeline =
        gst_rtsp_media_get_element(media);

    GstElement *appsrc =
        gst_bin_get_by_name_recurse_up(
            GST_BIN(pipeline),
            "mysrc");

    if (!appsrc)
    {
        logError("Cannot find appsrc element di pipeline media");
        gst_object_unref(pipeline);
        return;
    }

    GstCaps *caps =
        gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, self->m_width,
            "height", G_TYPE_INT, self->m_height,
            "framerate", GST_TYPE_FRACTION,
            self->m_fps, 1,
            nullptr);

    g_object_set(
        appsrc,
        "caps", caps,
        "format", GST_FORMAT_TIME,
        "is-live", TRUE,
        "block", FALSE,
        nullptr);

    gst_caps_unref(caps);

    {
        std::lock_guard<std::mutex> lock(self->m_appsrcMutex);
        self->m_appsrc = appsrc;
        self->m_appsrcReady = true;
    }
    self->m_appsrcCv.notify_all();

    logInfo("Client Connected");

    gst_object_unref(pipeline);
}

void RtspServer::needData(
    GstElement*,
    guint,
    gpointer)
{
}