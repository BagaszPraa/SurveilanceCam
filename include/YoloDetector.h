#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <array>

// Hasil satu deteksi
struct Detection {
    int classId;
    float confidence;
    cv::Rect box; // koordinat dalam skala frame asli
};

// Logger wajib untuk TensorRT (implementasi minimal, print warning/error ke stderr)
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// Deleter generik untuk objek TensorRT (dipakai unique_ptr)
struct TrtDestroy {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) delete obj;
    }
};

class YoloDetector {
public:
    YoloDetector(const std::string& enginePath,
                 int inputSize = 832,
                 float confThresh = 0.25f,
                 float nmsThresh = 0.45f,
                 std::vector<int> targetClasses = {});

    ~YoloDetector();

    std::vector<Detection> infer(const cv::Mat& frameBGR);

    void enableCuda(bool useCuda) {}

    std::string getClassName(int classId) const;
    bool loadClassNames(const std::string& filename);

    void setConfThreshold(float confThresh);
    float getConfThreshold() const;

    void setNmsThreshold(float nmsThresh);
    float getNmsThreshold() const;

    void setTargetClasses(const std::vector<int>& targetClasses);
    std::vector<int> getTargetClasses() const;
    void addTargetClass(int classId);
    void removeTargetClass(int classId);
    void clearTargetClasses();

    int getInputSize() const { return inputSize_; }
    int getInputWidth() const { return inputW_; }
    int getInputHeight() const { return inputH_; }
    int getInputChannels() const { return inputC_; }
    int getNumClasses() const { return static_cast<int>(classNames_.size()); }
    std::vector<std::string> getClassNames() const { return classNames_; }

private:
    static void logInfo(const std::string& msg);
    static void logWarn(const std::string& msg);
    static void logError(const std::string& msg);

private:
    std::vector<std::string> classNames_;
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime, TrtDestroy> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDestroy> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDestroy> context_;

    cudaStream_t stream_ = nullptr;
    cv::cuda::Stream cvStream_; // wrapper cv::cuda di atas stream_ yang sama

    std::string inputName_;
    std::string outputName_;

    int inputC_ = 3, inputH_ = 0, inputW_ = 0;
    int outputChannels_ = 0;
    int outputBoxes_ = 0;

    void* deviceInput_ = nullptr;
    void* deviceOutput_ = nullptr;

    // ---- Host output: pinned memory untuk transfer device->host lebih cepat ----
    // Input TIDAK lagi lewat host sama sekali -- preprocessing dilakukan
    // langsung di GPU dan hasilnya ditulis langsung ke deviceInput_.
    float* hostOutput_ = nullptr;
    size_t hostOutputSize_ = 0;

    // ---- Buffer GPU untuk preprocessing (dipakai ulang tiap frame, dialokasikan sekali) ----
    cv::cuda::GpuMat gpuFrame_;      // frame asli setelah upload
    cv::cuda::GpuMat gpuResized_;    // hasil resize (letterbox, sebelum padding)
    cv::cuda::GpuMat gpuLetterboxed_; // canvas penuh inputSize x inputSize (dengan padding)
    cv::cuda::GpuMat gpuRgb_;        // hasil convert BGR->RGB
    cv::cuda::GpuMat gpuFloat_;      // hasil convert ke float32 + normalisasi
    // 3 GpuMat yang masing-masing "membungkus" (wrap) alamat memory di
    // deviceInput_ per channel (CHW), supaya cv::cuda::split menulis
    // LANGSUNG ke buffer input TensorRT tanpa copy tambahan.
    std::array<cv::cuda::GpuMat, 3> gpuInputChannels_;

    int inputSize_;

    mutable std::mutex paramsMutex_;
    float confThresh_;
    float nmsThresh_;
    std::vector<int> targetClasses_;

    bool isTargetClass(int classId) const;
    void loadEngine(const std::string& enginePath);
    void allocateBuffers();
    // preprocess sekarang full GPU, output-nya langsung deviceInput_
    void preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY);
};