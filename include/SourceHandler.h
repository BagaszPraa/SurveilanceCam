#pragma once

#include <opencv2/opencv.hpp>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <condition_variable>
#include "ConfigManager.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <string>

class SourceHandler
{
public:
    bool openCapture(const Config& cfg, cv::VideoCapture& cap);
private:
    bool isNetworkStream(const std::string& src);
    std::string buildRtspGstPipeline(const std::string& url);

    // ---- Logging helper, prefix [rtspServer] ----
    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);

};