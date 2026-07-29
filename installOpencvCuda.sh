#!/bin/bash
# ============================================================
# installOpencvCuda.sh
# Build & install OpenCV dari source dengan CUDA support aktif
# (dibutuhkan agar cv2.cuda.getCudaEnabledDeviceCount() > 0)
#
# Syarat: jalankan install_deps.sh dulu (butuh NVIDIA driver + CUDA + cuDNN aktif)
# Estimasi waktu: 30-60 menit tergantung CPU
#
# PRASYARAT: opencv & opencv_contrib sudah ditambahkan sebagai submodule di repo ini:
#   git submodule add https://github.com/opencv/opencv.git deps/opencv
#   git submodule add https://github.com/opencv/opencv_contrib.git deps/opencv_contrib
#
# Jalankan (dari mana saja di dalam repo git, misal ~/SurveilanceCam):
#   sudo ./installOpencvCuda.sh
# ============================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if ! command -v nvcc &> /dev/null; then
    export PATH=/usr/local/cuda/bin:$PATH
fi
if ! command -v nvcc &> /dev/null; then
    echo -e "${YELLOW}nvcc tidak ditemukan. Pastikan CUDA toolkit sudah terinstall & reboot sudah dilakukan.${NC}"
    exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OPENCV_DIR="$REPO_ROOT/deps/opencv"
CONTRIB_DIR="$REPO_ROOT/deps/opencv_contrib"

echo -e "${GREEN}Build OpenCV dari submodule dengan CUDA...${NC}"
echo "Repo root : $REPO_ROOT"
echo "OpenCV    : $OPENCV_DIR"
echo "Contrib   : $CONTRIB_DIR"

# Isi submodule kalau masih kosong (misal repo baru di-clone tanpa --recurse-submodules)
if [ ! -f "$OPENCV_DIR/CMakeLists.txt" ] || [ ! -f "$CONTRIB_DIR/modules/README.md" ]; then
    echo -e "${YELLOW}Submodule belum terisi, menjalankan git submodule update --init --recursive...${NC}"
    cd "$REPO_ROOT"
    git submodule update --init --recursive deps/opencv deps/opencv_contrib
fi

if [ ! -f "$OPENCV_DIR/CMakeLists.txt" ]; then
    echo -e "${YELLOW}Submodule 'deps/opencv' tidak ditemukan. Tambahkan dulu:${NC}"
    echo "  git submodule add https://github.com/opencv/opencv.git deps/opencv"
    echo "  git submodule add https://github.com/opencv/opencv_contrib.git deps/opencv_contrib"
    exit 1
fi

CUDA_ARCH=$(python3 -c "
import subprocess
out = subprocess.run(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], capture_output=True, text=True)
print(out.stdout.strip().split('\n')[0] if out.stdout.strip() else '')
" 2>/dev/null)

cd "$OPENCV_DIR"

# Bersihkan folder build lama (kalau ada sisa dari percobaan sebelumnya yang gagal)
rm -rf build
mkdir -p build
cd build

# Deteksi versi CUDA toolkit yang terpasang, untuk info di log
CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
echo -e "${YELLOW}Terdeteksi CUDA toolkit versi: ${CUDA_VER}${NC}"

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH="$CONTRIB_DIR/modules" \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD_REQUIRED=ON \
    -D CUDA_NVCC_FLAGS="--std=c++17" \
    ${CUDA_ARCH:+-D CUDA_ARCH_BIN=$CUDA_ARCH} \
    ..

if ! make -j"$(nproc)"; then
    echo -e "${YELLOW}"
    echo "Build masih gagal. CUDA ${CUDA_VER} kemungkinan tidak cocok dengan versi OpenCV di submodule."
    echo "Opsi:"
    echo "  1) Ganti versi OpenCV di submodule, misal ke 4.14.0:"
    echo "     cd $OPENCV_DIR && git checkout 4.14.0 && cd $CONTRIB_DIR && git checkout 4.14.0"
    echo "     lalu jalankan ulang script ini."
    echo "  2) Atau downgrade ke CUDA 12.4 (lebih matang kompatibilitasnya):"
    echo "     sudo apt-get remove --purge 'cuda-*' 'libcudnn*' -y"
    echo "     sudo apt-get install -y cuda-toolkit-12-4"
    echo "     lalu jalankan ulang script ini."
    echo -e "${NC}"
    exit 1
fi
# make install
# ldconfig
echo "Jalankan perintah berikut untuk menginstall OpenCV ke sistem:"
echo "     sudo make install"
echo "     sudo ldconfig"
echo "python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep -A2 CUDA"