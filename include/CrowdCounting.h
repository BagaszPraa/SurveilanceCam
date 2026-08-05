#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

// Forward declare tipe TensorRT supaya header ini tidak perlu #include
// <NvInfer.h> (yang berat) -- cukup include di CrowdCounting.cpp.
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

// Hasil satu kali inferensi Crowd Counting.
// heatmapOverlay : cv::Mat BGR (ukuran sama dengan frame input) siap
//                   di-blend ke frame utama via cv::addWeighted.
// estimatedCount : estimasi jumlah orang pada frame tsb (integer, dibulatkan
//                   dari total density map).
// valid          : false kalau inferensi gagal / model belum siap.
struct CrowdCountResult {
    cv::Mat heatmapOverlay;
    int     estimatedCount = 0;
    bool    valid = false;
};

// Modul Crowd Counting berbasis density-map network (DM-Count, VGG-19
// backbone) yang sudah di-*build* jadi TensorRT engine FP32, mis. hasil:
//   trtexec --onnx=model_qnrf.onnx --saveEngine=model_qnrf.engine \
//           --minShapes=input:1x3x256x256 \
//           --optShapes=input:1x3x768x1024 \
//           --maxShapes=input:1x3x1536x2048
//
// PENTING soal file .engine:
// - File .engine bersifat spesifik terhadap GPU dan versi TensorRT yang
//   dipakai saat build. Engine yang di-build di satu device/versi TensorRT
//   TIDAK akan jalan di device/versi TensorRT lain -- harus di-rebuild
//   ulang di target device yang sebenarnya.
// - Kode ini ditulis untuk TensorRT 10+/11.x API (setTensorAddress by name +
//   enqueueV3).
// - Engine WAJIB bertipe FP32 murni di tensor input & output (dibuild dari
//   ONNX FP32 tanpa konversi manual ke FP16). loadEngine() akan memvalidasi
//   ini dan throw kalau ternyata engine-nya FP16/INT8.
//
// Thread-safety: satu instance CrowdCounting TIDAK boleh dipanggil infer()
// dari dua thread secara bersamaan (context TensorRT + buffer GPU dipakai
// bersama). Aman dipakai paralel BERSAMA YoloDetector karena keduanya
// instance terpisah. Di main.cpp setiap frame hanya memicu satu
// pemanggilan infer() via std::async yang selalu di-.get() sebelum frame
// berikutnya, jadi tidak ada pemanggilan konkuren ke instance yang sama.
class CrowdCounting {
public:
    // enginePath : path ke file .engine hasil build TensorRT (FP32).
    // inputSize  : ukuran (width, height) yang dipakai untuk inferensi.
    //              HARUS berada dalam rentang minShapes/maxShapes profile
    //              yang dipakai saat build engine. Default 1024x768 sesuai
    //              optShapes yang dipakai saat build
    //              (optShapes=input:1x3x768x1024 -> H=768, W=1024).
    CrowdCounting(const std::string& enginePath,
                  const cv::Size& inputSize = cv::Size(1024, 768));

    ~CrowdCounting();

    // Non-copyable (memegang handle GPU/CUDA)
    CrowdCounting(const CrowdCounting&) = delete;
    CrowdCounting& operator=(const CrowdCounting&) = delete;

    // Menjalankan inferensi pada satu frame BGR (hasil clone, bukan
    // referensi ke frame yang masih dipakai thread lain).
    CrowdCountResult infer(const cv::Mat& frame);

    void setHeatmapThreshold(float threshold) { heatmapThreshold_ = threshold; }

private:
    void loadEngine(const std::string& enginePath);
    void allocateBuffers();
    cv::Mat preprocess(const cv::Mat& frame) const;
    cv::Mat buildHeatmapOverlay(const cv::Mat& densityMap, const cv::Size& targetSize) const;

    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t                 stream_  = nullptr;

    // TensorRT 10+/11: binding tidak lagi pakai index, tapi nama tensor.
    std::string inputName_;
    std::string outputName_;
    // "density_map" -- output yang dipakai
    std::vector<std::string> extraOutputNames_;  // output lain yang wajib di-set address-nya tapi tidak dipakai
    std::vector<void*> extraOutputBuffers_;      // device buffer untuk extraOutputNames_

    void* deviceInputBuffer_  = nullptr;
    void* deviceOutputBuffer_ = nullptr;

    cv::Size inputSize_;      // width x height input ke model
    int      inputChannels_ = 3;
    int      outputHeight_  = 0; // diisi dari shape output engine
    int      outputWidth_   = 0;

    std::vector<float> hostInputBuffer_;
    std::vector<float> hostOutputBuffer_;

    float      heatmapThreshold_ = 1e-4f;
    std::mutex inferMutex_;
};