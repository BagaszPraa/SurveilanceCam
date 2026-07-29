#!/bin/bash
# ==========================================================
# Script konversi model YOLO (.pt) ke TensorRT (.engine)
# ==========================================================

# ---- KONFIGURASI ----
MODEL_PATH="models/best.pt"   # path model .pt yang mau dikonversi
IMG_SIZE=832                      # ukuran input image (samakan dengan saat inference)
HALF=true                         # true = pakai FP16 (lebih cepat, sedikit turun akurasi)
DYNAMIC=false                     # true jika ingin dynamic input shape
DEVICE=0                          # index GPU yang dipakai (0 = GPU pertama)
WORKSPACE=4                      # ukuran workspace TensorRT dalam GB

# ---- CEK APAKAH FILE MODEL ADA ----
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model tidak ditemukan di $MODEL_PATH"
    exit 1
fi

# ---- CEK APAKAH GPU TERSEDIA ----
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi tidak ditemukan. Pastikan driver NVIDIA & CUDA terinstall."
    exit 1
fi

echo "=========================================="
echo " Mulai konversi model ke TensorRT (.engine)"
echo " Model     : $MODEL_PATH"
echo " ImgSize   : $IMG_SIZE"
echo " Half(FP16): $HALF"
echo " Device    : GPU $DEVICE"
echo " Workspace : ${WORKSPACE}GB"
echo "=========================================="
echo " Catatan: proses build engine bisa memakan waktu"
echo " beberapa menit tergantung ukuran model & GPU."
echo "=========================================="

# ---- JALANKAN KONVERSI ----
yolo export \
    model="$MODEL_PATH" \
    format=engine \
    imgsz=$IMG_SIZE \
    half=$HALF \
    dynamic=$DYNAMIC \
    device=$DEVICE \
    workspace=$WORKSPACE

# ---- CEK HASIL ----
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo " Konversi berhasil!"
    echo " File .engine tersimpan di folder yang sama dengan model .pt"
    echo "=========================================="
else
    echo "=========================================="
    echo " Konversi GAGAL. Periksa error di atas."
    echo " Pastikan TensorRT & CUDA versi sudah sesuai"
    echo " dengan yang didukung ultralytics/torch kamu."
    echo "=========================================="
    exit 1
fi