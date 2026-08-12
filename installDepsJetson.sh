#!/bin/bash
# ============================================================
# install_deps_jetson.sh
# Auto-install semua dependensi untuk project SurveilanceCam
# (CUDA, cuDNN, TensorRT, OpenCV, GStreamer, dll)
#
# Setiap langkah SELALU dicek dulu — kalau komponen sudah
# terinstall (versi yang sesuai), langkah tersebut di-skip.
#
# ASUMSI: NVIDIA Jetson (ARM64) yang sudah di-flash dengan
# JetPack / L4T (Orin, Xavier, Nano, dll).
#
# Kalau ini PC/server dengan GPU NVIDIA biasa (x86_64),
# JANGAN pakai script ini — pakai install_deps.sh (versi desktop).
#
# CATATAN PENTING soal Jetson:
# - Driver GPU adalah bagian dari BSP/L4T (di-flash lewat
#   SDK Manager / jetpack image), BUKAN package yang diinstall
#   lewat apt seperti di PC biasa. Jadi TIDAK ADA langkah
#   "install nvidia driver" di script ini.
# - CUDA, cuDNN, TensorRT biasanya SUDAH ada kalau kamu flash
#   pakai JetPack image lengkap. Script ini hanya memverifikasi,
#   dan mencoba install lewat apt (repo NVIDIA L4T) kalau belum ada.
# - nvidia-smi TIDAK tersedia di Jetson — pakai tegrastats untuk
#   monitoring GPU.
# - PyTorch HARUS pakai wheel khusus Jetson dari NVIDIA
#   (bukan index PyPI/PyTorch biasa), karena arsitekturnya ARM64 + CUDA.
#
# Jalankan: sudo bash install_deps_jetson.sh
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
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }


# ============================================================
# -1. Pastikan ini benar-benar Jetson (bukan PC/x86_64)
# ============================================================
step "-1. DETEKSI PERANGKAT"
ARCH=$(uname -m)
if [ ! -f /etc/nv_tegra_release ] && [ "$ARCH" != "aarch64" ]; then
    fail "Perangkat ini sepertinya BUKAN Jetson (arch: $ARCH, /etc/nv_tegra_release tidak ada)."
    echo "  Gunakan install_deps.sh (versi desktop/x86) untuk PC dengan GPU NVIDIA biasa."
    exit 1
fi

if [ -f /etc/nv_tegra_release ]; then
    TEGRA_INFO=$(cat /etc/nv_tegra_release)
    ok "Jetson terdeteksi — $TEGRA_INFO"
else
    warn "Arch aarch64 terdeteksi tapi /etc/nv_tegra_release tidak ditemukan. Lanjut dengan hati-hati."
fi

if dpkg -s nvidia-jetpack &> /dev/null; then
    JETPACK_VER=$(dpkg -s nvidia-jetpack | grep '^Version' | cut -d' ' -f2)
    ok "JetPack meta-package terdeteksi: $JETPACK_VER"
else
    warn "Package nvidia-jetpack tidak terdeteksi. Kemungkinan L4T minimal (tanpa JetPack lengkap)."
fi

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
# 1. Cek driver/BSP L4T (bukan install — itu tugas SDK Manager)
# ============================================================
step "1. CEK DRIVER GPU (L4T/BSP)"
if [ -f /usr/lib/aarch64-linux-gnu/tegra/libcuda.so ] || [ -f /usr/lib/aarch64-linux-gnu/libnvidia-*.so* ] 2>/dev/null; then
    ok "Library NVIDIA/Tegra terdeteksi di sistem"
else
    warn "Tidak menemukan library Tegra GPU. Kalau ini Jetson asli, pastikan sudah di-flash dengan JetPack/SDK Manager terlebih dahulu — script ini TIDAK BISA menginstall driver GPU untuk Jetson."
fi
if command -v tegrastats &> /dev/null; then
    ok "tegrastats tersedia untuk monitoring GPU (pengganti nvidia-smi di Jetson)"
else
    warn "tegrastats tidak ditemukan (biasanya ada di /usr/bin/tegrastats)"
fi

# ============================================================
# 2. Repo NVIDIA L4T (biasanya sudah otomatis ada di image JetPack)
# ============================================================
step "2. CEK REPO NVIDIA L4T"
if [ -f /etc/apt/sources.list.d/nvidia-l4t-apt-source.list ]; then
    skip "Repo NVIDIA L4T sudah terpasang (nvidia-l4t-apt-source.list)"
    apt-get update -y
else
    fail "Repo NVIDIA L4T tidak ditemukan. Ini normal kalau bukan image resmi JetPack."
    echo "  CUDA/cuDNN/TensorRT mungkin tidak bisa diinstall otomatis lewat apt."
    echo "  Solusi: flash ulang board dengan SDK Manager, atau install komponen dari .deb resmi NVIDIA."
fi

# ============================================================
# 3. CUDA Toolkit
# ============================================================
step "3. CUDA TOOLKIT"
if command -v nvcc &> /dev/null; then
    skip "nvcc sudah ada ($(nvcc --version | grep release))"
else
    echo "nvcc tidak ditemukan, mencoba install cuda-toolkit via apt..."
    CUDA_PKG=$(apt-cache search '^cuda-toolkit-' 2>/dev/null | sort -V | tail -n1 | cut -d' ' -f1)
    if [ -n "$CUDA_PKG" ]; then
        apt-get install -y "$CUDA_PKG" || fail "Gagal install $CUDA_PKG. Di Jetson, CUDA biasanya HARUS lewat SDK Manager/JetPack, bukan apt manual."
    else
        fail "Tidak ada package cuda-toolkit-* yang ditemukan di repo. Install CUDA lewat NVIDIA SDK Manager (pilih komponen CUDA saat flashing/OTA)."
    fi
    if ! grep -q "cuda/bin" /etc/profile.d/cuda.sh 2>/dev/null; then
        cat > /etc/profile.d/cuda.sh << 'EOF'
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
        chmod +x /etc/profile.d/cuda.sh
    fi
    ok "CUDA Toolkit diproses (source /etc/profile.d/cuda.sh atau buka terminal baru)"
fi

# ============================================================
# 4. cuDNN
# ============================================================
step "4. cuDNN"
CUDNN_H=$(find /usr/include /usr/local/cuda*/include -name "cudnn_version.h" 2>/dev/null | head -n 1)
if [ -n "$CUDNN_H" ]; then
    skip "cuDNN header ditemukan: $CUDNN_H"
else
    apt-get install -y libcudnn9-dev 2>/dev/null \
        || apt-get install -y libcudnn8-dev 2>/dev/null \
        || fail "Gagal install cuDNN otomatis. Di Jetson, cuDNN biasanya HARUS dipilih lewat SDK Manager/JetPack, bukan apt manual."
    ok "cuDNN terinstall (atau sudah tersedia)"
fi

# ============================================================
# 5. TensorRT
# ============================================================
step "5. TensorRT"
if python3 -c "import tensorrt" &> /dev/null && command -v trtexec &> /dev/null; then
    skip "TensorRT sudah tersedia ($(python3 -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null))"
else
    apt-get install -y tensorrt python3-libnvinfer-dev python3-libnvinfer 2>/dev/null \
        || fail "Gagal install TensorRT via apt. Di Jetson, TensorRT normalnya sudah termasuk dalam image JetPack — cek lagi pilihan komponen saat flashing."

    # Kalau binding python belum kebaca di venv proyek, symlink dari dist-packages sistem
    if [ -n "$SUDO_USER" ]; then
        REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
        VENV_SITE="$REAL_HOME/SurveilanceCam/venv/lib/python3*/site-packages"
        SYS_TRT=$(python3 -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))" 2>/dev/null)
        if [ -n "$SYS_TRT" ]; then
            for target in $VENV_SITE; do
                if [ -d "$target" ] && [ ! -e "$target/tensorrt" ]; then
                    ln -s "$SYS_TRT" "$target/tensorrt" 2>/dev/null \
                        && ok "Symlink tensorrt dibuat ke venv: $target"
                fi
            done
        fi
    fi
    ok "TensorRT diproses (atau sudah tersedia)"
fi

# ============================================================
# 6. GStreamer
# ============================================================
step "6. GSTREAMER"
GST_PKGS="libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 libgstrtspserver-1.0-dev nvidia-l4t-gstreamer"
MISSING_GST=""
for pkg in $GST_PKGS; do
    dpkg -s "$pkg" &> /dev/null || MISSING_GST="$MISSING_GST $pkg"
done
if [ -z "$MISSING_GST" ]; then
    skip "Semua package GStreamer sudah terinstall"
else
    apt-get install -y $MISSING_GST || warn "Sebagian package GStreamer gagal diinstall (nvidia-l4t-gstreamer hanya ada di repo L4T resmi)"
    ok "GStreamer diproses:$MISSING_GST"
fi

if command -v gst-launch-1.0 &> /dev/null; then
    GST_VERSION=$(gst-launch-1.0 --version | head -n1)
    ok "GStreamer terdeteksi: $GST_VERSION"
    if gst-inspect-1.0 nvvidconv &> /dev/null; then
        ok "Plugin hardware-accel Jetson (nvvidconv) terdeteksi"
    else
        warn "Plugin nvvidconv/nvarguscamerasrc tidak terdeteksi — cek instalasi nvidia-l4t-gstreamer"
    fi
else
    fail "gst-launch-1.0 tidak ditemukan setelah instalasi"
fi

# ============================================================
# 7. OpenCV (Python)
# ============================================================
step "7. OPENCV (cek instalasi)"

if python3 -c "import cv2" &> /dev/null; then

    OPENCV_VERSION=$(python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null)
    CUDA_DEVICES=$(python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null)

    if [ -n "$CUDA_DEVICES" ] && [ "$CUDA_DEVICES" -gt 0 ] 2>/dev/null; then
        skip "OpenCV python sudah ada ($OPENCV_VERSION) — CUDA device terdeteksi: $CUDA_DEVICES"
    else
        skip "OpenCV python sudah ada ($OPENCV_VERSION) — TAPI CUDA device: 0 (kemungkinan build bawaan JetPack tanpa CUDA)"
    fi

    echo ""
    echo "Info build OpenCV (CUDA/cuDNN/GStreamer/Version control):"
    python3 -c "
import cv2
info = cv2.getBuildInformation()
for line in info.split('\n'):
    if any(k in line for k in ['CUDA', 'cuDNN', 'GStreamer', 'Version control']):
        print(line)
" 2>/dev/null || fail "Tidak bisa membaca build information OpenCV"

else

    echo -e "${RED}ERROR:${NC} OpenCV Python tidak ditemukan!"
    echo
    echo "Di Jetson, build OpenCV dengan CUDA dari source biasanya butuh script khusus"
    echo "(mis. dari JetsonHacks/Q-engineering — bukan skrip x86 installOpencvCuda.sh)."
    echo "Jalankan script build OpenCV-untuk-Jetson yang sesuai versi JetPack kamu."
    # exit 1

fi

# ============================================================
# 8. Python & Package AI/ML
# ============================================================
step "8. PYTHON PACKAGE AI/ML"

PKG_CHECK=$(python3 -c "
import importlib
pkgs = [
    ('pip', 'pip'),
    ('setuptools', 'setuptools'),
    ('wheel', 'wheel'),
    ('numpy', 'numpy'),
    ('h5py', 'h5py'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('ultralytics', 'ultralytics'),
    ('onnx', 'onnx'),
    ('onnxruntime', 'onnxruntime'),
    ('modelopt', 'nvidia-modelopt'),
]
missing = []
versions = []
for mod_name, disp_name in pkgs:
    try:
        m = importlib.import_module(mod_name)
        v = getattr(m, '__version__', 'unknown')
        versions.append(f'{disp_name} ({v})')
    except ImportError:
        missing.append(disp_name)
print('MISSING=' + ','.join(missing))
print('VERSIONS=' + ', '.join(versions))
")

MISSING_PKG=$(echo "$PKG_CHECK" | grep '^MISSING=' | cut -d'=' -f2)
INSTALLED_VER=$(echo "$PKG_CHECK" | grep '^VERSIONS=' | cut -d'=' -f2)

if [ -z "$MISSING_PKG" ]; then
    skip "Semua package sudah terinstall: $INSTALLED_VER"
else
    echo "Package belum lengkap, yang hilang: $MISSING_PKG"

    if echo "$MISSING_PKG" | grep -qE 'pip|setuptools|wheel'; then
        pip3 install --no-cache-dir --upgrade pip setuptools wheel
    fi

    if echo "$MISSING_PKG" | grep -qE 'torch\b|torchvision'; then
        warn "PyTorch di Jetson TIDAK BISA dari 'pip install torch' biasa maupun index cu124 (itu untuk x86_64)."
        echo "  Kamu harus install wheel resmi NVIDIA untuk Jetson (ARM64), sesuai versi JetPack yang terpasang."
        echo "  Cek versi JetPack kamu lalu ambil wheel yang cocok dari:"
        echo "    https://developer.download.nvidia.com/compute/redist/jp/"
        echo "  atau thread resmi PyTorch for Jetson di forum developer NVIDIA."
        echo "  Contoh pola (SESUAIKAN versi & JetPack, JANGAN asal copy):"
        echo "    pip3 install --no-cache-dir <URL_WHEEL_TORCH_JETSON_SESUAI_JETPACK>"
        echo "  torchvision biasanya perlu dibuild dari source agar cocok dengan versi torch Jetson tsb."
        echo "  Skip instalasi torch/torchvision otomatis — install manual sesuai panduan di atas."
    fi

    pip3 install --no-cache-dir \
        numpy \
        h5py \
        ultralytics \
        onnx \
        "nvidia-modelopt[onnx]>=0.44" \
        || warn "Sebagian package Python gagal diinstall — cek kompatibilitas versi untuk ARM64/Jetson"

    if echo "$MISSING_PKG" | grep -q 'onnxruntime'; then
        warn "onnxruntime-gpu dari PyPI biasa umumnya TIDAK build untuk Jetson."
        echo "  Ambil wheel onnxruntime-gpu khusus Jetson dari Jetson Zoo / NVIDIA Jetson AI Lab,"
        echo "  sesuai versi JetPack & CUDA yang terpasang."
    fi

    FINAL_VER=$(python3 -c "
import importlib
pkgs = [
    ('pip', 'pip'),
    ('setuptools', 'setuptools'),
    ('wheel', 'wheel'),
    ('numpy', 'numpy'),
    ('torch', 'torch'),
    ('torchvision', 'torchvision'),
    ('ultralytics', 'ultralytics'),
    ('onnx', 'onnx'),
    ('onnxruntime', 'onnxruntime'),
    ('modelopt', 'nvidia-modelopt'),
]
for mod_name, disp_name in pkgs:
    try:
        m = importlib.import_module(mod_name)
        v = getattr(m, '__version__', 'unknown')
        print(f'  {disp_name:<16}: {v}')
    except ImportError:
        print(f'  {disp_name:<16}: GAGAL DIIMPORT')
")
    echo ""
    echo "Versi terinstall:"
    echo "$FINAL_VER"

    ok "Package AI/ML diproses (lihat catatan di atas untuk torch/onnxruntime kalau perlu install manual)"
fi

# ============================================================
# Selesai
# ============================================================
separator
echo -e "${GREEN}Instalasi selesai untuk Jetson.${NC}"
echo -e "${YELLOW}Ingat:${NC} driver GPU, CUDA, cuDNN, dan TensorRT paling reliable didapat lewat"
echo "NVIDIA SDK Manager / OTA JetPack, bukan apt manual. Script ini hanya membantu"
echo "verifikasi dan melengkapi package tambahan (GStreamer, Python AI/ML)."
if [ "$REBOOT_NEEDED" -eq 1 ]; then
    echo -e "${RED}PENTING: Ada komponen sistem yang berubah. REBOOT sistem sekarang, lalu jalankan check_deps.sh untuk verifikasi.${NC}"
fi
separator