#!/bin/bash
# ============================================================
# check_deps.sh
# Cek semua versi dependensi untuk project SurveilanceCam
# (NVIDIA Driver, CUDA, cuDNN, TensorRT, OpenCV, GStreamer, dll)
# ============================================================

# Warna untuk output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

separator() {
    echo -e "${CYAN}------------------------------------------------------------${NC}"
}

check_header() {
    echo ""
    separator
    echo -e "${YELLOW}$1${NC}"
    separator
}

ok() {
    echo -e "${GREEN}✔${NC} $1"
}

fail() {
    echo -e "${RED}✘${NC} $1"
}

# ============================================================
# 1. OS Info
# ============================================================
check_header "1. OS INFO"
if command -v lsb_release &> /dev/null; then
    lsb_release -a 2>/dev/null | grep -E "Description|Release"
    ok "OS info terdeteksi"
else
    fail "lsb_release tidak ditemukan"
fi

# ============================================================
# 2. NVIDIA Driver & GPU
# ============================================================
check_header "2. NVIDIA DRIVER & GPU"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    ok "NVIDIA driver terdeteksi"
else
    fail "nvidia-smi tidak ditemukan — driver NVIDIA belum terinstall"
fi

# ============================================================
# 3. CUDA Toolkit
# ============================================================
check_header "3. CUDA TOOLKIT"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
    ok "nvcc terdeteksi di PATH"
else
    fail "nvcc tidak ditemukan di PATH (cek ~/.bashrc, export PATH cuda/bin)"
fi

echo ""
echo "CUDA Toolkit terinstall di sistem (/usr/local/):"
ls -d /usr/local/cuda-* 2>/dev/null || fail "Tidak ada folder /usr/local/cuda-* ditemukan"

# ============================================================
# 4. cuDNN
# ============================================================
check_header "4. cuDNN"
CUDNN_H=$(find /usr/include /usr/local/cuda*/include -name "cudnn_version.h" 2>/dev/null | head -n 1)
if [ -n "$CUDNN_H" ]; then
    grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" "$CUDNN_H"
    ok "cuDNN header ditemukan: $CUDNN_H"
else
    dpkg -l 2>/dev/null | grep -i cudnn || fail "cuDNN tidak ditemukan"
fi

# ============================================================
# 5. TensorRT
# ============================================================
check_header "5. TensorRT"
python3 -c "import tensorrt; print('TensorRT (python):', tensorrt.__version__)" 2>/dev/null \
    || fail "TensorRT python binding tidak ditemukan"

TRTEXEC_PATH=$(find / -name "trtexec" -type f 2>/dev/null | head -n 1)
if [ -n "$TRTEXEC_PATH" ]; then
    ok "trtexec ditemukan di: $TRTEXEC_PATH"
else
    fail "trtexec tidak ditemukan di sistem"
fi

echo ""
echo "Package TensorRT terinstall (dpkg):"
dpkg -l 2>/dev/null | grep -i tensorrt || fail "Tidak ada package tensorrt via dpkg"

# ============================================================
# 6. OpenCV
# ============================================================
check_header "6. OpenCV"
python3 -c "
import cv2
print('OpenCV version :', cv2.__version__)
print('CUDA device count :', cv2.cuda.getCudaEnabledDeviceCount())
" 2>/dev/null || fail "OpenCV python tidak ditemukan / gagal import"

# Cek build info detail (kalau ingin lihat lengkap: WITH_CUDA, GStreamer, dll)
echo ""
echo "Cari info build OpenCV (grep CUDA/GSTREAMER):"
python3 -c "
import cv2
info = cv2.getBuildInformation()
for line in info.split('\n'):
    if any(k in line for k in ['CUDA', 'cuDNN', 'GStreamer', 'Version control']):
        print(line)
" 2>/dev/null || fail "Tidak bisa membaca build information OpenCV"

# ============================================================
# 7. GStreamer
# ============================================================
check_header "7. GStreamer"
if command -v gst-launch-1.0 &> /dev/null; then
    gst-launch-1.0 --version
    ok "GStreamer terdeteksi"
else
    fail "gst-launch-1.0 tidak ditemukan"
fi

# ============================================================
# 8. Python & Package AI/ML
# ============================================================
check_header "8. PYTHON & PACKAGE AI/ML"
if command -v python3 &> /dev/null; then
    python3 --version
    ok "Python3 terdeteksi"
else
    fail "python3 tidak ditemukan"
fi

echo ""
for pkg in ultralytics torch onnx onnxruntime numpy; do
    VERSION=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
    if [ -n "$VERSION" ]; then
        ok "$pkg: $VERSION"
    else
        fail "$pkg belum terinstall"
    fi
done

# ============================================================
# 9. CMake & Compiler
# ============================================================
check_header "9. CMAKE & COMPILER"
command -v cmake &> /dev/null && cmake --version | head -n 1 || fail "cmake tidak ditemukan"
command -v g++ &> /dev/null && g++ --version | head -n 1 || fail "g++ tidak ditemukan"
command -v gcc &> /dev/null && gcc --version | head -n 1 || fail "gcc tidak ditemukan"

# ============================================================
# 10. Tesseract OCR (opsional, untuk ANPR)
# ============================================================
check_header "10. TESSERACT OCR (opsional)"
if command -v tesseract &> /dev/null; then
    tesseract --version | head -n 1
    ok "Tesseract terdeteksi"
else
    fail "Tesseract belum terinstall (opsional, hanya perlu kalau pakai OCR ini)"
fi

separator
echo -e "${GREEN}Selesai mengecek semua dependensi.${NC}"
separator