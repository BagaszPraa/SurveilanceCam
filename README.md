# gst_yolo_rtsp

Pipeline native **GStreamer C++**:

```
Input (USB UVC / RTSP) → capture (appsink) → inferensi YOLO (OpenCV DNN) → output RTSP server (appsrc → x264enc → rtph264pay)
```
## 1. Buat VENV & Install YOLO

```bash
python3 -m venv venv
source /path/to/venv/bin/activate

git submodule update --init --recursive

pip install ultralytics numpy

# pastikan venv aktif dulu

pip install --upgrade pip
pip install --extra-index-url https://pypi.nvidia.com tensorrt
```

## 1. Dependency

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake pkg-config \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libopencv-dev
```

> Pastikan paket `libgstrtspserver-1.0-dev` terinstall — ini yang menyediakan `gstreamer-rtsp-server-1.0` untuk RTSP server native.

**Tambahan wajib untuk backend TensorRT:**
- **CUDA Toolkit** (driver NVIDIA + `nvcc`) — biasanya sudah ada kalau di Jetson (JetPack) atau habis install driver GPU + CUDA di PC.
- **TensorRT** — di Jetson/JetPack biasanya sudah bundled. Di PC/server, install lewat `apt` (`nvidia-tensorrt` / paket dari NVIDIA repo) atau download tar.gz dari NVIDIA Developer.

Cek ketersediaan:
```bash
nvcc --version
dpkg -l | grep nvinfer
```

## 2. Siapkan model (.engine)

Model harus format **`.engine`** (TensorRT), hasil dari:
```bash
yolo export model=visDrone.pt format=engine imgsz=832 half=true device=0
```

```bash
mkdir -p models
# taruh visDrone.engine hasil export di sini
```

> **PENTING:** file `.engine` **terikat ke GPU & versi TensorRT** tempat dia dibuild. Kalau kamu pindah device (misal dari PC dev ke Jetson, atau ganti GPU), **wajib build ulang** `.engine` di device target — tidak portable seperti `.onnx`.

Kalau `CMakeLists.txt` tidak otomatis menemukan TensorRT (error `Tidak menemukan libnvinfer`), set manual:
```bash
cmake -DTENSORRT_DIR=/path/ke/TensorRT-x.y.z ..
```

## 3. Build

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Output binary: `build/gst_yolo_rtsp`

## 4. Jalankan

**Input dari USB UVC camera:**
```bash
./gst_yolo_rtsp \
    --input-type usb \
    --input /dev/video0 \
    --model ../models/visDrone.engine \
    --width 1280 --height 720 \
    --port 8554 --mount /stream
```

**Input dari RTSP camera (misal CCTV IP):**
```bash
./gst_yolo_rtsp \
    --input-type rtsp \
    --input rtsp://user:pass@192.168.1.10:554/stream1 \
    --model ../models/visDrone.engine \
    --width 1280 --height 720 \
    --port 8554 --mount /stream
```

## 5. Lihat hasil output RTSP

Dari device lain di jaringan yang sama, buka pakai VLC atau ffplay:

```bash
ffplay rtsp://<IP_SERVER>:8554/stream
```

atau

```bash
vlc rtsp://<IP_SERVER>:8554/stream
```

## Argumen CLI

| Argumen         | Default              | Keterangan                                  |
|-----------------|----------------------|----------------------------------------------|
| `--input-type`  | `usb`                | `usb` atau `rtsp`                             |
| `--input`       | `/dev/video0`        | path device UVC atau URL RTSP                 |
| `--model`       | `models/visDrone.engine` | path model TensorRT (.engine)            |
| `--width`       | `1280`                | lebar frame capture & output                  |
| `--height`      | `720`                 | tinggi frame capture & output                 |
| `--infer-size`  | `832`                 | ukuran input model (samakan dengan saat export ONNX) |
| `--conf`        | `0.25`                | confidence threshold                          |
| `--port`        | `8554`                | port RTSP server                              |
| `--mount`       | `/stream`             | mount point RTSP (jadi bagian URL)            |

## Catatan performa

- `x264enc speed-preset=ultrafast tune=zerolatency` dipakai untuk minim latency, cocok untuk surveillance real-time. Bisa diganti `nvh264enc` kalau GPU NVIDIA & plugin `gstreamer1.0-plugins-bad` dengan NVENC tersedia (jauh lebih cepat, encode di GPU).
- Kelas target default hanya `person` (`classId 0`, lihat `Config::targetClasses` di `main.cpp`). Ubah sesuai kebutuhan (misal tambah `2,3,5,7` untuk car/motorcycle/bus/truck).
- Filter bbox digambar **tanpa label kelas/confidence**, sesuai preferensi tampilan bersih. Kalau mau matikan overlay counter juga, hapus bagian `cv::putText` di `main.cpp`.