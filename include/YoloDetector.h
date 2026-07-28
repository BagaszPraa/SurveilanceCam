#pragma once

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>
#include <memory>
#include <mutex>

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
    // enginePath    : path ke model .engine (hasil `yolo export format=engine`)
    // inputSize     : ukuran input model (harus sama dengan saat export, misal 832)
    // confThresh    : ambang confidence
    // nmsThresh     : ambang NMS
    // targetClasses : filter class id (kosong = semua class)
    YoloDetector(const std::string& enginePath,
                 int inputSize = 832,
                 float confThresh = 0.25f,
                 float nmsThresh = 0.45f,
                 std::vector<int> targetClasses = {});

    ~YoloDetector();

    // Jalankan inferensi pada satu frame BGR
    std::vector<Detection> infer(const cv::Mat& frameBGR);

    // Tidak dipakai untuk TensorRT (selalu GPU), disediakan agar kompatibel
    // dengan pemanggilan lama (no-op).
    void enableCuda(bool useCuda) {}

    std::string getClassName(int classId) const;
    bool loadClassNames(const std::string& filename);

    // -----------------------------------------------------------
    // Setting parameter deteksi objek (lengkap, thread-safe)
    // -----------------------------------------------------------
    // Semua setter di bawah aman dipanggil dari thread lain saat infer()
    // sedang berjalan di thread lain (dilindungi m_paramsMutex). Perubahan
    // akan langsung berlaku pada pemanggilan infer() berikutnya.

    // Ambang confidence minimum supaya deteksi dianggap valid
    void setConfThreshold(float confThresh);
    float getConfThreshold() const;

    // Ambang IoU untuk Non-Maximum Suppression
    void setNmsThreshold(float nmsThresh);
    float getNmsThreshold() const;

    // Filter class id yang ingin dideteksi (kosong = semua class)
    void setTargetClasses(const std::vector<int>& targetClasses);
    std::vector<int> getTargetClasses() const;
    void addTargetClass(int classId);
    void removeTargetClass(int classId);
    void clearTargetClasses(); // kosongkan filter -> deteksi semua class kembali

    // Info-info bawaan model (read-only, ditentukan saat engine dimuat)
    int getInputSize() const { return inputSize_; }
    int getInputWidth() const { return inputW_; }
    int getInputHeight() const { return inputH_; }
    int getInputChannels() const { return inputC_; }
    int getNumClasses() const { return static_cast<int>(classNames_.size()); }
    std::vector<std::string> getClassNames() const { return classNames_; }

private:
    // -----------------------------------------------------------
    // Logging, prefix [YoloDetector]
    // -----------------------------------------------------------
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

    // TensorRT 10.x/11.x sudah tidak punya binding index (getTensorShape(int),
    // enqueueV2, dst). Semua tensor I/O diakses lewat NAMA, bukan index.
    std::string inputName_;
    std::string outputName_;

    // Dimensi tensor (diambil dari engine)
    int inputC_ = 3, inputH_ = 0, inputW_ = 0;
    int outputChannels_ = 0; // 4 + num_classes
    int outputBoxes_ = 0;

    // Buffer device terpisah (bukan array binding index lagi), didaftarkan
    // ke context lewat setTensorAddress() saat allocateBuffers()
    void* deviceInput_ = nullptr;
    void* deviceOutput_ = nullptr;

    std::vector<float> hostInput_;
    std::vector<float> hostOutput_;

    int inputSize_;

    // Parameter deteksi -- bisa diubah runtime lewat setter di atas,
    // dilindungi mutex supaya aman dipanggil dari thread lain.
    mutable std::mutex paramsMutex_;
    float confThresh_;
    float nmsThresh_;
    std::vector<int> targetClasses_;

    bool isTargetClass(int classId) const;
    void loadEngine(const std::string& enginePath);
    void allocateBuffers();
    void preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY);
};