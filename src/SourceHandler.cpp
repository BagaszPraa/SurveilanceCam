#include "SourceHandler.h"
#include "ConfigManager.h"
#include <algorithm>
#include <cctype>

// ---------------------------------------------------
// Logging
// ---------------------------------------------------
void SourceHandler::logInfo(const std::string& msg)
{
    std::cout << "[SourceHandler] [INFO] " << msg << std::endl;
}

void SourceHandler::logWarn(const std::string& msg)
{
    std::cout << "[SourceHandler] [WARN] " << msg << std::endl;
}

void SourceHandler::logError(const std::string& msg)
{
    std::cerr << "[SourceHandler] [ERROR] " << msg << std::endl;
}

// ---------------------------------------------------
// Cek apakah sebuah GStreamer element factory benar-benar ADA
// di sistem ini. Ini kunci "auto" yang sesungguhnya: bukan nebak
// dari nama platform, tapi tanya langsung ke GStreamer registry.
// ---------------------------------------------------
bool SourceHandler::gstElementExists(const std::string& factoryName)
{
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }
    GstElementFactory* f = gst_element_factory_find(factoryName.c_str());
    if (f) {
        gst_object_unref(f);
        return true;
    }
    return false;
}

// ---------------------------------------------------
// Prioritas auto-select decoder:
//   1. Jetson       -> nvv4l2decoder (L4T multimedia API, dipakai baik utk H264 maupun H265)
//   2. NVIDIA dGPU  -> nvh264dec / nvh265dec (plugin nvcodec, gstreamer1.0-plugins-bad)
//   3. CPU          -> avdec_h264 / avdec_h265 (software, libav)
// ---------------------------------------------------
GstDecoderBackend SourceHandler::detectDecoderBackend(VideoCodecType codec)
{
    if (gstElementExists("nvv4l2decoder")) {
        return GstDecoderBackend::JETSON;
    }

    const std::string nvElem = (codec == VideoCodecType::H265) ? "nvh265dec" : "nvh264dec";
    if (gstElementExists(nvElem)) {
        return GstDecoderBackend::NVIDIA_DGPU;
    }

    const std::string avElem = (codec == VideoCodecType::H265) ? "avdec_h265" : "avdec_h264";
    if (gstElementExists(avElem)) {
        return GstDecoderBackend::CPU;
    }

    return GstDecoderBackend::UNKNOWN;
}

// ---------------------------------------------------
// Probe codec RTSP (H264/H265) pakai GstDiscoverer sebelum pipeline
// dibangun, supaya depay/parse element yang dipasang sesuai (rtph264depay
// vs rtph265depay tidak bisa dipakai tertukar). Ada timeout biar gak
// nge-hang kalau stream mati/lambat -> fallback ke H264.
// ---------------------------------------------------
VideoCodecType SourceHandler::probeRtspCodec(const std::string& url, int timeoutSec)
{
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    GError* err = nullptr;
    GstDiscoverer* discoverer = gst_discoverer_new((GstClockTime)timeoutSec * GST_SECOND, &err);
    if (!discoverer) {
        logWarn("Gagal membuat GstDiscoverer: " + std::string(err ? err->message : "unknown") +
                " -> fallback H264");
        if (err) g_error_free(err);
        return VideoCodecType::H264;
    }

    GError* discErr = nullptr;
    GstDiscovererInfo* info = gst_discoverer_discover_uri(discoverer, url.c_str(), &discErr);
    VideoCodecType result = VideoCodecType::UNKNOWN;

    if (!info || discErr != nullptr) {
        logWarn("Probe codec RTSP gagal/timeout, fallback ke H264. (" +
                std::string(discErr ? discErr->message : "no info") + ")");
        if (discErr) g_error_free(discErr);
    } else {
        GList* streams = gst_discoverer_info_get_video_streams(info);
        for (GList* it = streams; it != nullptr; it = it->next) {
            GstDiscovererStreamInfo* sinfo = (GstDiscovererStreamInfo*)it->data;
            GstCaps* caps = gst_discoverer_stream_info_get_caps(sinfo);
            if (caps) {
                gchar* capsStr = gst_caps_to_string(caps);
                std::string s(capsStr ? capsStr : "");
                std::transform(s.begin(), s.end(), s.begin(), ::tolower);

                if (s.find("video/x-h265") != std::string::npos) {
                    result = VideoCodecType::H265;
                } else if (s.find("video/x-h264") != std::string::npos) {
                    result = VideoCodecType::H264;
                }

                if (capsStr) g_free(capsStr);
                gst_caps_unref(caps);
            }
            if (result != VideoCodecType::UNKNOWN) break;
        }
        if (streams) gst_discoverer_stream_info_list_free(streams);
    }

    if (info) gst_discoverer_info_unref(info);
    g_object_unref(discoverer);

    if (result == VideoCodecType::UNKNOWN) {
        logWarn("Codec tidak terdeteksi dari stream, default ke H264");
        result = VideoCodecType::H264;
    }
    return result;
}

std::string SourceHandler::codecToString(VideoCodecType c)
{
    switch (c) {
        case VideoCodecType::H264: return "H264";
        case VideoCodecType::H265: return "H265";
        default: return "UNKNOWN";
    }
}

std::string SourceHandler::backendToString(GstDecoderBackend b)
{
    switch (b) {
        case GstDecoderBackend::JETSON:      return "JETSON (nvv4l2decoder)";
        case GstDecoderBackend::NVIDIA_DGPU: return "NVIDIA dGPU (nvcodec)";
        case GstDecoderBackend::CPU:         return "CPU (avdec, software)";
        default:                             return "UNKNOWN";
    }
}

std::string SourceHandler::joinPipeline(const std::vector<std::string>& elements)
{
    std::string result;
    for (size_t i = 0; i < elements.size(); ++i) {
        result += elements[i];
        if (i + 1 < elements.size()) result += " ! ";
    }
    return result;
}

// ---------------------------------------------------
// Capture
// ---------------------------------------------------
bool SourceHandler::openCapture(const Config& cfg, cv::VideoCapture& cap)
{
    bool isNumericIndex = !cfg.inputSource.empty() &&
        std::all_of(cfg.inputSource.begin(), cfg.inputSource.end(), ::isdigit);

    if (cfg.isGstreamer) {
        std::string gstPipeline = isNetworkStream(cfg.inputSource)
            ? buildRtspGstPipeline(cfg)   // user cuma kasih URL polos -> auto-build pipeline
            : cfg.inputSource;            // user sudah kasih pipeline GStreamer lengkap -> pakai apa adanya
        logInfo("GStreamer pipeline: " + gstPipeline);
        cap.open(gstPipeline, cv::CAP_GSTREAMER);
    } else if (isNumericIndex) {
        // USB webcam via index
        cap.open(std::stoi(cfg.inputSource), cv::CAP_V4L2);
    } else {
        // path device (/dev/video0) atau file video lokal
        cap.open(cfg.inputSource, cv::CAP_V4L2);
    }

    if (!cap.isOpened()) {
        logError("Failed to open video capture");
        return false;
    }

    if (!cfg.isGstreamer) {
        // Properti ini cuma relevan untuk device kamera lokal (V4L2).
        // Ini hanya "permintaan" ke driver -- resolusi FINAL tetap dideteksi
        // ulang dari frame asli (probe frame), karena driver/kamera bisa
        // saja tidak persis menuruti angka yang diminta.
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
        cap.set(cv::CAP_PROP_FPS, cfg.fps);
    }

    return true;
}

bool SourceHandler::isNetworkStream(const std::string& src)
{
    return src.rfind("rtsp://", 0) == 0 ||
           src.rfind("rtmp://", 0) == 0 ||
           src.rfind("http://", 0) == 0 ||
           src.rfind("https://", 0) == 0;
}

// ---------------------------------------------------
// Bangun pipeline GStreamer utk RTSP secara otomatis:
//   1. Probe codec (H264/H265) via GstDiscoverer
//   2. Pilih decoder terbaik yang TERSEDIA di sistem (Jetson > NVIDIA dGPU > CPU)
//   3. Susun rantai depay/parse/decode sesuai hasil deteksi
//   4. Resize (kalau cfg.width & cfg.height > 0) pakai elemen yang cocok
//      dengan backend-nya (nvvidconv utk Jetson, videoscale utk CPU/dGPU)
//
// appsink: drop=true + max-buffers=1 + sync=false -> selalu ambil frame
// TERBARU (latest-frame), bukan antre, cocok buat pipeline low-latency.
// ---------------------------------------------------
std::string SourceHandler::buildRtspGstPipeline(const Config& cfg)
{
    const std::string url = cfg.inputSource;

    VideoCodecType codec = probeRtspCodec(url);
    GstDecoderBackend backend = detectDecoderBackend(codec);

    logInfo("Codec RTSP terdeteksi : " + codecToString(codec));
    logInfo("Decoder backend dipilih: " + backendToString(backend));

    const bool doResize = (cfg.width > 0 && cfg.height > 0);
    const std::string wStr = std::to_string(cfg.width);
    const std::string hStr = std::to_string(cfg.height);

    const std::string depay = (codec == VideoCodecType::H265) ? "rtph265depay" : "rtph264depay";
    const std::string parse = (codec == VideoCodecType::H265) ? "h265parse" : "h264parse";

    std::vector<std::string> chain;
    chain.push_back("rtspsrc location=" + url + " latency=0 protocols=tcp");
    chain.push_back(depay);
    chain.push_back(parse);

    switch (backend) {
        case GstDecoderBackend::JETSON: {
            // nvv4l2decoder -> output NVMM memory, WAJIB lewat nvvidconv
            // dulu sebelum bisa dipakai OpenCV. Resize sekalian dititipkan
            // di caps setelah nvvidconv (paling efisien, resize di GPU).
            chain.push_back("nvv4l2decoder");
            chain.push_back("nvvidconv");
            chain.push_back(doResize
                ? "video/x-raw,format=BGRx,width=" + wStr + ",height=" + hStr
                : "video/x-raw,format=BGRx");
            chain.push_back("videoconvert");
            chain.push_back("video/x-raw,format=BGR");
            break;
        }
        case GstDecoderBackend::NVIDIA_DGPU: {
            // nvh264dec/nvh265dec (plugin nvcodec). videoscale dipasang
            // sebelum videoconvert supaya resize terjadi sebelum konversi
            // colorspace final ke BGR.
            chain.push_back((codec == VideoCodecType::H265) ? "nvh265dec" : "nvh264dec");
            if (doResize) chain.push_back("videoscale");
            chain.push_back("videoconvert");
            chain.push_back(doResize
                ? "video/x-raw,format=BGR,width=" + wStr + ",height=" + hStr
                : "video/x-raw,format=BGR");
            break;
        }
        case GstDecoderBackend::CPU:
        default: {
            // avdec_h264/avdec_h265 (software, libav) sebagai fallback
            // paling universal kalau tidak ada hardware decoder.
            chain.push_back((codec == VideoCodecType::H265) ? "avdec_h265" : "avdec_h264");
            if (doResize) chain.push_back("videoscale");
            chain.push_back("videoconvert");
            chain.push_back(doResize
                ? "video/x-raw,format=BGR,width=" + wStr + ",height=" + hStr
                : "video/x-raw,format=BGR");
            break;
        }
    }

    chain.push_back("appsink drop=true max-buffers=1 sync=false");

    return joinPipeline(chain);
}