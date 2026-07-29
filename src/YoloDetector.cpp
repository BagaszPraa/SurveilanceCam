#include "YoloDetector.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// ==========================================================
// Helper macro cek error CUDA
// ==========================================================
#define CUDA_CHECK(status)                                                     \
    do {                                                                       \
        auto ret = (status);                                                   \
        if (ret != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                      cudaGetErrorString(ret));                 \
        }                                                                       \
    } while (0)

// ==========================================================
// Logging, prefix [YoloDetector]
// ==========================================================
void YoloDetector::logInfo(const std::string& msg) {
    std::cout << "[YoloDetector] [INFO] " << msg << std::endl;
}

void YoloDetector::logWarn(const std::string& msg) {
    std::cout << "[YoloDetector] [WARN] " << msg << std::endl;
}

void YoloDetector::logError(const std::string& msg) {
    std::cerr << "[YoloDetector] [ERROR] " << msg << std::endl;
}

// ==========================================================
// TrtLogger
// ==========================================================
// Pesan dari TensorRT sendiri (bukan dari kode YoloDetector) tetap disalurkan
// lewat helper log di atas supaya format & prefix-nya konsisten.
void TrtLogger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[YoloDetector] [ERROR] [TensorRT] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "[YoloDetector] [WARN] [TensorRT] " << msg << std::endl;
            break;
        case Severity::kINFO:
        case Severity::kVERBOSE:
        default:
            // INFO/VERBOSE dari TensorRT cukup berisik kalau selalu dicetak,
            // jadi sengaja diabaikan di sini (hanya WARNING ke atas yang
            // ditampilkan). Ubah kalau butuh log TensorRT lebih detail.
            break;
    }
}
// ==========================================================
// Constructor / Destructor
// ==========================================================
YoloDetector::YoloDetector(const std::string& enginePath,
                            int inputSize,
                            float confThresh,
                            float nmsThresh,
                            std::vector<int> targetClasses)
    : inputSize_(inputSize),
      confThresh_(confThresh),
      nmsThresh_(nmsThresh),
      targetClasses_(std::move(targetClasses)) {
    logInfo("Menginisialisasi detector, engine: " + enginePath);

    std::string classFile = enginePath;
    size_t pos = classFile.rfind(".engine");
    if(pos != std::string::npos)
    {
        classFile.replace(pos, 7, ".names");
    }
    loadClassNames(classFile);
    loadEngine(enginePath);
    allocateBuffers();

    CUDA_CHECK(cudaStreamCreate(&stream_));

    logInfo("Detector siap. confThresh=" + std::to_string(confThresh_) +
            ", nmsThresh=" + std::to_string(nmsThresh_) +
            ", targetClasses=" + std::to_string(targetClasses_.size()) +
            " (0 = semua class)");
}

YoloDetector::~YoloDetector() {
    if (deviceInput_) cudaFree(deviceInput_);
    if (deviceOutput_) cudaFree(deviceOutput_);
    if (hostOutput_) cudaFreeHost(hostOutput_);
    if (stream_) cudaStreamDestroy(stream_);
    logInfo("Detector dihentikan, resource dibersihkan.");
}

std::string YoloDetector::getClassName(int classId) const
{
    if (classId >= 0 && classId < static_cast<int>(classNames_.size()))
        return classNames_[classId];

    return std::to_string(classId);
}

bool YoloDetector::loadClassNames(const std::string& filename)
{
    classNames_.clear();

    std::ifstream file(filename);

    if (!file.is_open())
    {
        logWarn("Tidak bisa membuka file class names: " + filename +
                " (deteksi tetap jalan, tapi label class akan ditampilkan sebagai angka)");
        return false;
    }

    std::string line;

    while (std::getline(file, line))
    {
        if (!line.empty())
            classNames_.push_back(line);
    }

    logInfo("Berhasil memuat " + std::to_string(classNames_.size()) + " class dari: " + filename);

    return true;
}

// ==========================================================
// Setting parameter deteksi objek (thread-safe)
// ==========================================================
void YoloDetector::setConfThreshold(float confThresh) {
    if (confThresh < 0.0f || confThresh > 1.0f) {
        logWarn("setConfThreshold() diabaikan: nilai " + std::to_string(confThresh) +
                " di luar rentang valid [0,1].");
        return;
    }
    std::lock_guard<std::mutex> lock(paramsMutex_);
    confThresh_ = confThresh;
    logInfo("confThresh diubah menjadi " + std::to_string(confThresh_));
}

float YoloDetector::getConfThreshold() const {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    return confThresh_;
}

void YoloDetector::setNmsThreshold(float nmsThresh) {
    if (nmsThresh < 0.0f || nmsThresh > 1.0f) {
        logWarn("setNmsThreshold() diabaikan: nilai " + std::to_string(nmsThresh) +
                " di luar rentang valid [0,1].");
        return;
    }
    std::lock_guard<std::mutex> lock(paramsMutex_);
    nmsThresh_ = nmsThresh;
    logInfo("nmsThresh diubah menjadi " + std::to_string(nmsThresh_));
}

float YoloDetector::getNmsThreshold() const {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    return nmsThresh_;
}

void YoloDetector::setTargetClasses(const std::vector<int>& targetClasses) {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    targetClasses_ = targetClasses;
    logInfo("targetClasses diganti, jumlah class difilter: " + std::to_string(targetClasses_.size()) +
            (targetClasses_.empty() ? " (kosong = semua class)" : ""));
}

std::vector<int> YoloDetector::getTargetClasses() const {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    return targetClasses_;
}

void YoloDetector::addTargetClass(int classId) {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    if (std::find(targetClasses_.begin(), targetClasses_.end(), classId) == targetClasses_.end()) {
        targetClasses_.push_back(classId);
        logInfo("Class id " + std::to_string(classId) + " ditambahkan ke targetClasses.");
    } else {
        logWarn("Class id " + std::to_string(classId) + " sudah ada di targetClasses, diabaikan.");
    }
}

void YoloDetector::removeTargetClass(int classId) {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    auto it = std::find(targetClasses_.begin(), targetClasses_.end(), classId);
    if (it != targetClasses_.end()) {
        targetClasses_.erase(it);
        logInfo("Class id " + std::to_string(classId) + " dihapus dari targetClasses.");
    } else {
        logWarn("Class id " + std::to_string(classId) + " tidak ditemukan di targetClasses, diabaikan.");
    }
}

void YoloDetector::clearTargetClasses() {
    std::lock_guard<std::mutex> lock(paramsMutex_);
    targetClasses_.clear();
    logInfo("targetClasses dikosongkan, sekarang mendeteksi semua class.");
}

// ==========================================================
// Load & deserialize engine dari file .engine
// ==========================================================
void YoloDetector::loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        logError("Tidak bisa buka file engine: " + enginePath);
        throw std::runtime_error("Tidak bisa buka file engine: " + enginePath +
                                  " (cek path, relatif terhadap folder tempat binary dijalankan)");
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // ---- Skip metadata JSON yang disisipkan Ultralytics di awal file ----
    // File .engine hasil `yolo export format=engine` BUKAN raw TensorRT plan.
    // Formatnya: [4 byte int32 = panjang metadata][metadata JSON][engine TRT asli]
    // Metadata berisi info seperti nama class, stride, imgsz, dll (dipakai
    // ultralytics Python, tidak kita perlukan di C++). Kalau bagian ini tidak
    // di-skip, deserializeCudaEngine akan membaca JSON sebagai header TensorRT
    // dan gagal di pengecekan magicTag walaupun engine-nya valid.
    int32_t metaLen = 0;
    file.read(reinterpret_cast<char*>(&metaLen), sizeof(metaLen));
    if (!file.good() || metaLen < 0 ||
        static_cast<std::streamsize>(sizeof(metaLen) + metaLen) >= fileSize) {
        logError("Gagal membaca header metadata pada file engine: " + enginePath);
        throw std::runtime_error(
            "Gagal membaca header metadata pada file engine: " + enginePath +
            " (format file tidak sesuai dugaan; pastikan file ini benar hasil "
            "`yolo export format=engine`, bukan raw TensorRT plan)");
    }
    file.seekg(metaLen, std::ios::cur); // lompati metadata JSON

    std::streamsize engineByteSize = fileSize - static_cast<std::streamsize>(sizeof(metaLen)) - metaLen;
    std::vector<char> engineData(engineByteSize);
    if (!file.read(engineData.data(), engineByteSize)) {
        logError("Gagal membaca isi engine (setelah metadata) dari file: " + enginePath);
        throw std::runtime_error("Gagal membaca isi engine (setelah metadata) dari file: " + enginePath);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        logError("Gagal membuat TensorRT IRuntime");
        throw std::runtime_error("Gagal membuat TensorRT IRuntime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineByteSize));
    if (!engine_) {
        logError("Gagal deserialize engine: " + enginePath);
        throw std::runtime_error(
            "Gagal deserialize engine: " + enginePath +
            " (kemungkinan versi TensorRT saat build engine beda dengan versi runtime saat ini, "
            "atau engine dibuild di GPU/arsitektur berbeda -- build ulang .engine di device ini)");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        logError("Gagal membuat TensorRT IExecutionContext");
        throw std::runtime_error("Gagal membuat TensorRT IExecutionContext");
    }

    // ---- Cari nama tensor input & output (API name-based, TensorRT 10.x/11.x) ----
    // Index-based binding (getTensorShape(int), enqueueV2, dst) sudah DIHAPUS
    // dari TensorRT 10.x ke atas. Semua akses sekarang lewat nama tensor.
    int nbTensors = engine_->getNbIOTensors();
    bool foundInput = false, foundOutput = false;

    for (int i = 0; i < nbTensors; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
            // Dims biasanya [1, 3, H, W]
            inputC_ = dims.d[1];
            inputH_ = dims.d[2];
            inputW_ = dims.d[3];
            foundInput = true;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
            // Dims biasanya [1, 4+num_classes, num_boxes]
            outputChannels_ = dims.d[1];
            outputBoxes_ = dims.d[2];
            foundOutput = true;
        }
    }

    if (!foundInput || !foundOutput) {
        logError("Tidak bisa menemukan tensor input/output pada engine: " + enginePath);
        throw std::runtime_error("Tidak bisa menemukan tensor input/output pada engine. "
                                  "Cek apakah engine hasil export YOLO standar (1 input, 1 output).");
    }

    // Jika engine dibuild dengan dynamic shape (dim = -1, misal batch dinamis),
    // set shape eksplisit sebelum inference. Untuk kasus umum (batch=1 statis)
    // ini tidak wajib, tapi aman untuk dipanggil sekali di sini.
    if (engine_->getTensorShape(inputName_.c_str()).d[0] == -1) {
        context_->setInputShape(inputName_.c_str(),
                                 nvinfer1::Dims4{1, inputC_, inputH_, inputW_});
    }

    logInfo("Engine loaded. Input '" + inputName_ + "': " +
            std::to_string(inputC_) + "x" + std::to_string(inputH_) + "x" + std::to_string(inputW_) +
            " | Output '" + outputName_ + "' channels: " + std::to_string(outputChannels_) +
            ", boxes: " + std::to_string(outputBoxes_));
}

// ==========================================================
// Alokasi buffer GPU (device) + host untuk input/output
// ==========================================================
void YoloDetector::allocateBuffers() {
    size_t inputVolume = static_cast<size_t>(inputC_) * inputH_ * inputW_;
    size_t outputVolume = static_cast<size_t>(outputChannels_) * outputBoxes_;

    hostOutputSize_ = outputVolume;

    // Host output pakai pinned memory -> cudaMemcpyAsync device->host benar-benar async
    CUDA_CHECK(cudaHostAlloc(&hostOutput_, outputVolume * sizeof(float), cudaHostAllocDefault));

    CUDA_CHECK(cudaMalloc(&deviceInput_, inputVolume * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&deviceOutput_, outputVolume * sizeof(float)));

    context_->setTensorAddress(inputName_.c_str(), deviceInput_);
    context_->setTensorAddress(outputName_.c_str(), deviceOutput_);

    // ---- Bungkus 3 GpuMat, masing-masing menunjuk ke bagian channel yang
    //      berbeda di deviceInput_ (layout CHW: channel 0, lalu channel 1, dst).
    //      cv::cuda::split() nanti menulis LANGSUNG ke sini -- tidak ada
    //      copy tambahan device->device maupun device->host->device. ----
    size_t channelBytes = static_cast<size_t>(inputH_) * inputW_ * sizeof(float);
    for (int c = 0; c < 3; ++c) {
        float* channelPtr = reinterpret_cast<float*>(
            reinterpret_cast<char*>(deviceInput_) + c * channelBytes);
        gpuInputChannels_[c] = cv::cuda::GpuMat(inputH_, inputW_, CV_32FC1, channelPtr);
    }

    // Wrapper cv::cuda::Stream di atas stream_ yang sama, supaya semua
    // operasi cv::cuda:: dan cudaMemcpyAsync/enqueueV3 tetap satu urutan
    // (sequential) di stream CUDA yang sama -- tidak perlu sync manual
    // antara preprocessing dan inferensi.
    cvStream_ = cv::cuda::StreamAccessor::wrapStream(stream_);

    logInfo("Buffer GPU dialokasikan (pinned host output + GPU preprocessing). "
            "Input volume: " + std::to_string(inputVolume) +
            " elemen, Output volume: " + std::to_string(outputVolume) + " elemen.");
}

bool YoloDetector::isTargetClass(int classId) const {
    // Dipanggil dari dalam infer() yang sudah memegang lock paramsMutex_
    // sendiri (lihat infer()), jadi tidak lock lagi di sini untuk hindari
    // deadlock/rekursif lock.
    if (targetClasses_.empty()) return true;
    return std::find(targetClasses_.begin(), targetClasses_.end(), classId) != targetClasses_.end();
}

// ==========================================================
// Preprocessing GPU: upload -> letterbox (resize+pad) -> BGR2RGB
// -> normalize -> split channel LANGSUNG ke deviceInput_ (CHW)
// Tidak ada data yang mampir ke host sama sekali di sini.
// ==========================================================
void YoloDetector::preprocess(const cv::Mat& frame, float& scale, int& padX, int& padY) {
    if (frame.empty()) {
        logError("preprocess() menerima frame kosong, inferensi dibatalkan.");
        throw std::runtime_error("YoloDetector::preprocess: frame kosong");
    }

    int w = frame.cols;
    int h = frame.rows;

    scale = std::min(static_cast<float>(inputSize_) / w, static_cast<float>(inputSize_) / h);
    int newW = static_cast<int>(w * scale);
    int newH = static_cast<int>(h * scale);
    padX = (inputSize_ - newW) / 2;
    padY = (inputSize_ - newH) / 2;

    // ---- Upload frame asli (BGR, CV_8UC3) ke GPU ----
    gpuFrame_.upload(frame, cvStream_);

    // ---- Resize ke ukuran target (sebelum padding) ----
    cv::cuda::resize(gpuFrame_, gpuResized_, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR, cvStream_);

    // ---- Siapkan canvas penuh inputSize x inputSize, isi warna padding 114 ----
    if (gpuLetterboxed_.empty() ||
        gpuLetterboxed_.size() != cv::Size(inputSize_, inputSize_) ||
        gpuLetterboxed_.type() != frame.type()) {
        gpuLetterboxed_.create(inputSize_, inputSize_, frame.type());
    }
    gpuLetterboxed_.setTo(cv::Scalar(114, 114, 114), cvStream_);

    // ---- Copy hasil resize ke tengah canvas (area letterbox) ----
    cv::cuda::GpuMat roi = gpuLetterboxed_(cv::Rect(padX, padY, newW, newH));
    gpuResized_.copyTo(roi, cvStream_);

    // ---- BGR -> RGB ----
    cv::cuda::cvtColor(gpuLetterboxed_, gpuRgb_, cv::COLOR_BGR2RGB, 0, cvStream_);

    // ---- uint8 -> float32, normalisasi [0,255] -> [0,1] ----
    gpuRgb_.convertTo(gpuFloat_, CV_32FC3, 1.0 / 255.0, 0.0, cvStream_);

    // ---- HWC -> CHW: split channel LANGSUNG ke deviceInput_ (via GpuMat wrapper) ----
    cv::cuda::split(gpuFloat_, gpuInputChannels_.data(), cvStream_);

    // Catatan: tidak ada cudaStreamSynchronize di sini secara sengaja --
    // semua operasi di atas dan enqueueV3() nanti dijalankan berurutan
    // di stream CUDA yang sama (stream_), jadi urutannya tetap terjamin
    // tanpa perlu blocking manual di titik ini.
}
// ==========================================================
// Inferensi utama
// ==========================================================
std::vector<Detection> YoloDetector::infer(const cv::Mat& frameBGR) {
    float scale = 1.0f;
    int padX = 0, padY = 0;

    // preprocess() sekarang menulis LANGSUNG ke deviceInput_ via GPU,
    // jadi tidak ada lagi cudaMemcpyAsync host->device di sini.
    preprocess(frameBGR, scale, padX, padY);

    float confThreshSnapshot;
    float nmsThreshSnapshot;
    std::vector<int> targetClassesSnapshot;
    {
        std::lock_guard<std::mutex> lock(paramsMutex_);
        confThreshSnapshot = confThresh_;
        nmsThreshSnapshot = nmsThresh_;
        targetClassesSnapshot = targetClasses_;
    }

    // ---- Jalankan inferensi (sequential di stream yang sama dengan preprocessing) ----
    bool ok = context_->enqueueV3(stream_);
    if (!ok) {
        logError("TensorRT enqueueV3 gagal dijalankan");
        throw std::runtime_error("TensorRT enqueueV3 gagal dijalankan");
    }

    // ---- Copy output device -> host (pinned memory, benar-benar async) ----
    CUDA_CHECK(cudaMemcpyAsync(hostOutput_, deviceOutput_,
                                hostOutputSize_ * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_));

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // ---- Decode output (sama seperti sebelumnya, tidak berubah) ----
    int numClasses = outputChannels_ - 4;

    auto isTargetClassLocal = [&targetClassesSnapshot](int classId) -> bool {
        if (targetClassesSnapshot.empty()) return true;
        return std::find(targetClassesSnapshot.begin(), targetClassesSnapshot.end(), classId)
               != targetClassesSnapshot.end();
    };

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int b = 0; b < outputBoxes_; ++b) {
        float cx = hostOutput_[0 * outputBoxes_ + b];
        float cy = hostOutput_[1 * outputBoxes_ + b];
        float w  = hostOutput_[2 * outputBoxes_ + b];
        float h  = hostOutput_[3 * outputBoxes_ + b];

        int bestClassId = -1;
        float bestScore = 0.0f;
        for (int c = 0; c < numClasses; ++c) {
            float score = hostOutput_[(4 + c) * outputBoxes_ + b];
            if (score > bestScore) {
                bestScore = score;
                bestClassId = c;
            }
        }

        if (bestScore < confThreshSnapshot) continue;
        if (!isTargetClassLocal(bestClassId)) continue;

        float x1 = (cx - w / 2.0f - padX) / scale;
        float y1 = (cy - h / 2.0f - padY) / scale;
        float boxW = w / scale;
        float boxH = h / scale;

        boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1),
                            static_cast<int>(boxW), static_cast<int>(boxH));
        scores.push_back(bestScore);
        classIds.push_back(bestClassId);
    }

    std::vector<int> keepIndices;
    cv::dnn::NMSBoxes(boxes, scores, confThreshSnapshot, nmsThreshSnapshot, keepIndices);

    std::vector<Detection> results;
    results.reserve(keepIndices.size());
    for (int idx : keepIndices) {
        Detection det;
        det.classId = classIds[idx];
        det.confidence = scores[idx];
        det.box = boxes[idx] & cv::Rect(0, 0, frameBGR.cols, frameBGR.rows);
        results.push_back(det);
    }

    return results;
}