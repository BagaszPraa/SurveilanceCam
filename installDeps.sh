#!/bin/bash
# ============================================================
# install_deps.sh
# Auto-install semua dependensi untuk project SurveilanceCam
# (NVIDIA Driver, CUDA, cuDNN, TensorRT, OpenCV, GStreamer, dll)
#
# Setiap langkah SELALU dicek dulu — kalau komponen sudah
# terinstall (versi yang sesuai), langkah tersebut di-skip.
#
# ASUMSI: Ubuntu x86_64 dengan GPU NVIDIA (BUKAN Jetson/ARM).
# Kalau ini Jetson, JANGAN pakai script ini — pakai JetPack/SDK Manager.
#
# Jalankan: sudo bash install_deps.sh
# ============================================================

set -e  # stop kalau ada command yang gagal

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

separator() { echo -e "${CYAN}------------------------------------------------------------${NC}"; }
step() { echo ""; separator; echo -e "${YELLOW}$1${NC}"; separator; }
ok() { echo -e "${GREEN}✔${NC} $1"; }
fail() { echo -e "${RED}✘${NC} $1"; }
skip() { echo -e "${CYAN}⏭${NC}  $1 — skip install"; }


REBOOT_NEEDED=0

# ============================================================
# 0. Update sistem & base tools
# ============================================================
step "0. UPDATE SYSTEM & INSTALL BASE TOOLS"
BASE_PKGS="build-essential cmake g++ gcc git wget curl software-properties-common ca-certificates gnupg lsb-release pkg-config unzip python3 python3-pip python3-dev python3-venv"
MISSING_BASE=""
for pkg in $BASE_PKGS; do
    dpkg -s "$pkg" &> /dev/null || MISSING_BASE="$MISSING_BASE $pkg"
done
if [ -z "$MISSING_BASE" ]; then
    skip "Semua base tools sudah terinstall"
else
    apt-get update -y
    apt-get install -y --no-install-recommends $MISSING_BASE
    ok "Base tools terinstall:$MISSING_BASE"
fi

# ============================================================
# 1. NVIDIA Driver
# ============================================================
step "1. NVIDIA DRIVER"
if command -v nvidia-smi &> /dev/null; then
    skip "nvidia-smi sudah ada ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1))"
else
    echo "Driver NVIDIA belum ditemukan, menginstall via ubuntu-drivers..."
    apt-get install -y ubuntu-drivers-common
    ubuntu-drivers autoinstall
    REBOOT_NEEDED=1
    ok "Driver NVIDIA terinstall — WAJIB REBOOT sebelum lanjut ke langkah CUDA/cuDNN/TensorRT"
fi

# ============================================================
# 2. NVIDIA CUDA repo (dipakai untuk CUDA, cuDNN, TensorRT)
# ============================================================
step "2. SETUP REPO NVIDIA CUDA"
if [ -f /usr/share/keyrings/cuda-archive-keyring.gpg ] || dpkg -l | grep -q cuda-keyring; then
    skip "Repo NVIDIA CUDA sudah terpasang"
else
    UBUNTU_VER=$(lsb_release -rs | tr -d '.')
    DISTRO="ubuntu${UBUNTU_VER}"
    cd /tmp
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-keyring_1.1-1_all.deb" \
        -O cuda-keyring.deb || fail "Gagal download keyring, cek koneksi/versi Ubuntu (${DISTRO})"
    if [ -f cuda-keyring.deb ]; then
        dpkg -i cuda-keyring.deb
        apt-get update -y
        ok "Repo NVIDIA CUDA terpasang"
    fi
fi

# ============================================================
# 3. CUDA Toolkit
# ============================================================
step "3. CUDA TOOLKIT"
if command -v nvcc &> /dev/null; then
    skip "nvcc sudah ada ($(nvcc --version | grep release))"
else
    apt-get install -y cuda-toolkit-13-3 || fail "Gagal install cuda-toolkit-13-3, cek nama package yang tersedia: apt-cache search cuda-toolkit"
    # Tambahkan CUDA ke PATH untuk semua user
    if ! grep -q "cuda/bin" /etc/profile.d/cuda.sh 2>/dev/null; then
        cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
        chmod +x /etc/profile.d/cuda.sh
    fi
    ok "CUDA Toolkit terinstall (source /etc/profile.d/cuda.sh atau buka terminal baru)"
fi

# ============================================================
# 4. cuDNN
# ============================================================
step "4. cuDNN"
CUDNN_H=$(find /usr/include /usr/local/cuda*/include -name "cudnn_version.h" 2>/dev/null | head -n 1)
if [ -n "$CUDNN_H" ]; then
    skip "cuDNN header ditemukan: $CUDNN_H"
else
    apt-get install -y cudnn9-cuda-12 2>/dev/null \
        || apt-get install -y libcudnn9-dev-cuda-12 2>/dev/null \
        || fail "Gagal install cuDNN otomatis. Cek nama package: apt-cache search cudnn"
    ok "cuDNN terinstall"
fi

# ============================================================
# 5. TensorRT
# ============================================================
step "5. TensorRT"
if python3 -c "import tensorrt" &> /dev/null && command -v trtexec &> /dev/null; then
    skip "TensorRT sudah tersedia ($(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null))"
else
    apt-get install -y tensorrt python3-libnvinfer-dev 2>/dev/null \
        || fail "Gagal install TensorRT via apt. Cek: apt-cache search tensorrt"
    pip3 install --no-cache-dir tensorrt 2>/dev/null \
        || echo "  (python binding TensorRT biasanya ikut dari apt package di atas)"
    ok "TensorRT terinstall (atau sudah tersedia)"
fi

# ============================================================
# 6. GStreamer
# ============================================================
step "6. GSTREAMER"
GST_PKGS="libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 libgstrtspserver-1.0-dev"
MISSING_GST=""
for pkg in $GST_PKGS; do
    dpkg -s "$pkg" &> /dev/null || MISSING_GST="$MISSING_GST $pkg"
done
if [ -z "$MISSING_GST" ]; then
    skip "Semua package GStreamer sudah terinstall"
else
    apt-get install -y $MISSING_GST
    ok "GStreamer terinstall:$MISSING_GST"
fi

# ============================================================
# 7. OpenCV (Python)
# ============================================================

step "7. OPENCV (cek instalasi)"

if python3 -c "import cv2" &> /dev/null; then

    OPENCV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
    skip "OpenCV python sudah ada ($OPENCV_VERSION)"

else

    echo -e "${RED}ERROR:${NC} OpenCV Python tidak ditemukan!"
    echo
    echo "Install OpenCV dari source dengan CUDA support:"
    echo
    echo "1. Jalankan script:"
    echo "   ./installOpencvCuda.sh"
    # exit 1

fi

echo -e "${YELLOW}Catatan:${NC}"
echo "OpenCV dari pip (opencv-python) tidak memiliki cv2.cuda."
echo "Gunakan build source dengan CUDA untuk mendapatkan CUDA acceleration."

# ============================================================
# 8. Python & Package AI/ML
# ============================================================
step "8. PYTHON PACKAGE AI/ML"
pip3 install --no-cache-dir --upgrade pip setuptools wheel

check_pkg() { python3 -c "import $1" &> /dev/null; }

if check_pkg numpy && check_pkg torch && check_pkg torchvision && check_pkg ultralytics && check_pkg onnx && check_pkg onnxruntime; then
    skip "numpy, torch, torchvision, ultralytics, onnx, onnxruntime sudah terinstall"
else
    pip3 install --no-cache-dir \
        numpy \
        ultralytics 
        # torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
        # onnx \
        # onnxruntime-gpu
    ok "Package AI/ML terinstall"
fi

# ============================================================
# 9. Tesseract OCR (opsional, untuk ANPR)
# ============================================================
step "9. TESSERACT OCR"
if command -v tesseract &> /dev/null; then
    skip "Tesseract sudah ada ($(tesseract --version | head -n1))"
else
    apt-get install -y tesseract-ocr libtesseract-dev
    ok "Tesseract terinstall"
fi

# ============================================================
# Selesai
# ============================================================
separator
echo -e "${GREEN}Instalasi selesai.${NC}"
if [ "$REBOOT_NEEDED" -eq 1 ]; then
    echo -e "${RED}PENTING: Driver NVIDIA baru diinstall. REBOOT sistem sekarang, lalu jalankan check_deps.sh untuk verifikasi.${NC}"
fi
echo "Jalankan check_deps.sh untuk verifikasi semua komponen."
separator