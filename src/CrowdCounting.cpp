#include "CrowdCounting.h"

#include <NvInfer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace {

void logInfo(const std::string& msg) {
    std::cout << "[CrowdCounting] [INFO] " << msg << std::endl;
}
void logWarn(const std::string& msg) {
    std::cout << "[CrowdCounting] [WARN] " << msg << std::endl;
}
void logError(const std::string& msg) {
    std::cerr << "[CrowdCounting] [ERROR] " << msg << std::endl;
}

// Logger minimal untuk TensorRT. Semua pesan level WARNING ke atas
// diteruskan ke stderr, INFO/VERBOSE diabaikan supaya log tidak berisik.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

TrtLogger& trtLogger() {
    static TrtLogger instance;
    return instance;
}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t status = (call);                                              \
        if (status != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +                \
                                      cudaGetErrorString(status) +                 \
                                      " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                          \
    } while (0)

} // namespace

CrowdCounting::CrowdCounting(const std::string& enginePath,
                              const cv::Size& inputSize,
                              bool useFp16)
    : inputSize_(inputSize), useFp16_(useFp16) {
    loadEngine(enginePath);
    allocateBuffers();

    logInfo("Engine dimuat: " + enginePath +
            " | input: " + std::to_string(inputSize_.width) + "x" + std::to_string(inputSize_.height) +
            " | output density map: " + std::to_string(outputWidth_) + "x" + std::to_string(outputHeight_) +
            " | precision: " + std::string(useFp16_ ? "FP16" : "FP32 (asumsi, sesuai build engine)"));
}

CrowdCounting::~CrowdCounting() {
    if (deviceInputBuffer_)  cudaFree(deviceInputBuffer_);
    if (deviceOutputBuffer_) cudaFree(deviceOutputBuffer_);
    if (stream_)             cudaStreamDestroy(stream_);

    // TensorRT 10+/11: destroy() sudah dihapus dari API, sekarang objek
    // di-delete langsung (destructor virtual sudah disediakan TensorRT).
    delete context_;
    delete engine_;
    delete runtime_;
}

void CrowdCounting::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("File engine tidak ditemukan/tidak bisa dibuka: " + enginePath);
    }

    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engineData(static_cast<size_t>(size));
    if (!file.read(engineData.data(), size)) {
        throw std::runtime_error("Gagal membaca file engine: " + enginePath);
    }

    runtime_ = nvinfer1::createInferRuntime(trtLogger());
    if (!runtime_) {
        throw std::runtime_error("Gagal membuat TensorRT runtime.");
    }

    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_) {
        throw std::runtime_error("Gagal deserialize engine: " + enginePath +
                                  " (kemungkinan engine di-build dengan versi TensorRT/GPU berbeda).");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("Gagal membuat execution context dari engine.");
    }

    // ---- Cari nama tensor input & output ----
    // TensorRT 10+/11: tidak ada lagi binding index, semua diakses lewat
    // nama tensor. Asumsi: engine punya tepat 1 input dan 1 output
    // (density map). Kalau modelmu multi-output, sesuaikan di sini.
    const int nbIO = engine_->getNbIOTensors();
    for (int i = 0; i < nbIO; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
        } else {
            outputName_ = name;
        }
    }

    if (inputName_.empty() || outputName_.empty()) {
        throw std::runtime_error("Tidak bisa menemukan tensor input/output pada engine.");
    }

    // ---- Set shape input kalau engine pakai dynamic shape ----
    nvinfer1::Dims inputDims = engine_->getTensorShape(inputName_.c_str());
    // Format umum NCHW: [batch, channel, H, W]. Dimensi bernilai -1 berarti
    // dynamic, ditimpa dengan inputSize_ yang dikonfigurasi.
    if (inputDims.nbDims == 4) {
        if (inputDims.d[0] < 0) inputDims.d[0] = 1; // batch = 1
        if (inputDims.d[2] < 0) inputDims.d[2] = inputSize_.height;
        if (inputDims.d[3] < 0) inputDims.d[3] = inputSize_.width;
        inputChannels_ = (inputDims.d[1] > 0) ? inputDims.d[1] : 3;

        if (!context_->setInputShape(inputName_.c_str(), inputDims)) {
            throw std::runtime_error("setInputShape gagal -- cek profile shape engine cocok dengan inputSize.");
        }
    } else {
        throw std::runtime_error("Bentuk dimensi input engine tidak dikenali (harap NCHW rank 4).");
    }

    // ---- Ambil shape output setelah shape input di-set ----
    nvinfer1::Dims outputDims = context_->getTensorShape(outputName_.c_str());
    if (outputDims.nbDims < 2) {
        throw std::runtime_error("Bentuk dimensi output engine tidak dikenali.");
    }
    outputHeight_ = outputDims.d[outputDims.nbDims - 2];
    outputWidth_  = outputDims.d[outputDims.nbDims - 1];
}

void CrowdCounting::allocateBuffers() {
    CUDA_CHECK(cudaStreamCreate(&stream_));

    const size_t inputElements  = static_cast<size_t>(inputChannels_) * inputSize_.height * inputSize_.width;
    const size_t outputElements = static_cast<size_t>(outputHeight_) * outputWidth_;

    hostInputBuffer_.resize(inputElements);
    hostOutputBuffer_.resize(outputElements);

    CUDA_CHECK(cudaMalloc(&deviceInputBuffer_,  inputElements  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceOutputBuffer_, outputElements * sizeof(float)));
}

cv::Mat CrowdCounting::preprocess(const cv::Mat& frame) const {
    cv::Mat resized;
    cv::resize(frame, resized, inputSize_, 0, 0, cv::INTER_LINEAR);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    cv::Mat floatImg;
    rgb.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // Normalisasi ImageNet-style, umum dipakai backbone VGG/ResNet pada
    // density-map network. Sesuaikan kalau model kamu pakai normalisasi lain.
    cv::subtract(floatImg, cv::Scalar(0.485, 0.456, 0.406), floatImg);
    cv::divide(floatImg, cv::Scalar(0.229, 0.224, 0.225), floatImg);

    return floatImg; // HWC, float32
}

CrowdCountResult CrowdCounting::infer(const cv::Mat& frame) {
    CrowdCountResult result;

    if (frame.empty()) {
        logWarn("Menerima frame kosong, inferensi dilewati.");
        return result;
    }

    std::lock_guard<std::mutex> lock(inferMutex_);

    try {
        cv::Mat floatImg = preprocess(frame);

        // ---- HWC -> CHW ke host buffer ----
        std::vector<cv::Mat> channels(3);
        cv::split(floatImg, channels);
        const size_t planeSize = static_cast<size_t>(inputSize_.height) * inputSize_.width;
        for (int c = 0; c < inputChannels_; ++c) {
            std::memcpy(hostInputBuffer_.data() + c * planeSize,
                        channels[c].ptr<float>(),
                        planeSize * sizeof(float));
        }

        // ---- Host -> Device, enqueue inferensi, Device -> Host ----
        CUDA_CHECK(cudaMemcpyAsync(deviceInputBuffer_, hostInputBuffer_.data(),
                                    hostInputBuffer_.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_));

        // TensorRT 10+/11: tidak ada lagi array bindings by index. Setiap
        // tensor I/O di-bind lewat namanya masing-masing, lalu dijalankan
        // dengan enqueueV3 (stream saja, tanpa parameter bindings).
        if (!context_->setTensorAddress(inputName_.c_str(), deviceInputBuffer_)) {
            throw std::runtime_error("setTensorAddress gagal untuk tensor input: " + inputName_);
        }
        if (!context_->setTensorAddress(outputName_.c_str(), deviceOutputBuffer_)) {
            throw std::runtime_error("setTensorAddress gagal untuk tensor output: " + outputName_);
        }

        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("enqueueV3 gagal dijalankan.");
        }

        CUDA_CHECK(cudaMemcpyAsync(hostOutputBuffer_.data(), deviceOutputBuffer_,
                                    hostOutputBuffer_.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream_));

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        cv::Mat densityMap(outputHeight_, outputWidth_, CV_32F, hostOutputBuffer_.data());

        double totalDensity = cv::sum(densityMap)[0];
        result.estimatedCount = static_cast<int>(std::round(std::max(0.0, totalDensity)));

        result.heatmapOverlay = buildHeatmapOverlay(densityMap, frame.size());
        result.valid = true;

    } catch (const std::exception& e) {
        logError("Inferensi gagal: " + std::string(e.what()));
        result.valid = false;
    }

    return result;
}

cv::Mat CrowdCounting::buildHeatmapOverlay(const cv::Mat& densityMap, const cv::Size& targetSize) const {
    cv::Mat clipped;
    cv::threshold(densityMap, clipped, heatmapThreshold_, 0.0, cv::THRESH_TOZERO);

    double minVal, maxVal;
    cv::minMaxLoc(clipped, &minVal, &maxVal);

    cv::Mat normalized;
    if (maxVal > 1e-6) {
        clipped.convertTo(normalized, CV_8U, 255.0 / maxVal);
    } else {
        normalized = cv::Mat::zeros(clipped.size(), CV_8U);
    }

    cv::Mat colorMap;
    cv::applyColorMap(normalized, colorMap, cv::COLORMAP_JET);

    cv::Mat resized;
    cv::resize(colorMap, resized, targetSize, 0, 0, cv::INTER_LINEAR);

    cv::Mat mask;
    cv::resize(normalized, mask, targetSize, 0, 0, cv::INTER_LINEAR);
    cv::Mat maskedOverlay = cv::Mat::zeros(resized.size(), resized.type());
    resized.copyTo(maskedOverlay, mask);

    return maskedOverlay;
}