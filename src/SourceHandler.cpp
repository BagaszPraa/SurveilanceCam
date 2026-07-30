#include "SourceHandler.h"
#include "ConfigManager.h"


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

bool SourceHandler::openCapture(const Config& cfg, cv::VideoCapture& cap) {
    bool isNumericIndex = !cfg.inputSource.empty() &&
        std::all_of(cfg.inputSource.begin(), cfg.inputSource.end(), ::isdigit);

    if (cfg.isGstreamer) {
        std::string gstPipeline = isNetworkStream(cfg.inputSource)
            ? buildRtspGstPipeline(cfg.inputSource)   // user cuma kasih URL polos -> auto-build pipeline
            : cfg.inputSource;                        // user sudah kasih pipeline GStreamer lengkap -> pakai apa adanya
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
// tambahkan kembali helper ini
bool SourceHandler::isNetworkStream(const std::string& src) {
    return src.rfind("rtsp://", 0) == 0 ||
           src.rfind("rtmp://", 0) == 0 ||
           src.rfind("http://", 0) == 0 ||
           src.rfind("https://", 0) == 0;
}

// Pipeline GStreamer untuk RTSP input via OpenCV appsink.
// latency=0 supaya jitter buffer rtspsrc minim (low-latency).
// videoconvert + BGR karena OpenCV cv::Mat default-nya BGR.
// drop=true + max-buffers=1 di appsink supaya selalu ambil frame TERBARU,
// bukan antre (mirip pola "latest frame" yang sudah dipakai di capture thread).
std::string SourceHandler::buildRtspGstPipeline(const std::string& url) {
    return
        "rtspsrc location=" + url + " latency=0 protocols=tcp "
        "! rtph264depay "
        "! h264parse "
        "! avdec_h264 "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink drop=true max-buffers=1 sync=false";
}