# GStreamer RTSP Server (C++)

RTSP Server berbasis GStreamer yang mendukung multiple stream dengan encoding H.264.

## Fitur

| Mount Point | Deskripsi | Resolusi | Codec |
|-------------|-----------|----------|-------|
| `/test`     | Video test pattern (bouncing ball) + audio tone 440Hz | 640×480 | H.264 + AAC |
| `/camera`   | Webcam live `/dev/video0` | 640×480 | H.264 |
| `/hd`       | HD test pattern, ultra-low latency | 1280×720 | H.264 |
| `/file`     | Stream dari file video (opsional) | original | H.264 + AAC |

## Dependensi

```bash
# Ubuntu / Debian
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstrtspserver-1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    cmake build-essential

# Fedora / RHEL
sudo dnf install -y \
    gstreamer1-devel \
    gstreamer1-rtsp-server-devel \
    gstreamer1-plugins-base-devel
```

## Build

### Menggunakan Makefile
```bash
make
```

### Menggunakan CMake
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Penggunaan

```bash
# Jalankan dengan port default (8554)
./rtsp_server

# Dengan port kustom
./rtsp_server 9000

# Dengan file video tambahan
./rtsp_server 8554 /path/to/video.mp4
```

## Membuka Stream

### FFplay (ffmpeg)
```bash
ffplay rtsp://localhost:8554/test
ffplay rtsp://localhost:8554/hd
```

### VLC
```bash
vlc rtsp://localhost:8554/test
```

### GStreamer (gst-launch)
```bash
gst-launch-1.0 rtspsrc location=rtsp://localhost:8554/test ! \
    rtph264depay ! h264parse ! avdec_h264 ! autovideosink
```

### OpenCV (Python)
```python
import cv2
cap = cv2.VideoCapture("rtsp://localhost:8554/test")
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("RTSP Stream", frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

## Arsitektur Pipeline

```
GStreamer RTSP Server
│
├── /test  → videotestsrc → x264enc → rtph264pay (pay0)
│           audiotestsrc → avenc_aac → rtpmp4apay (pay1)
│
├── /camera → v4l2src → x264enc → rtph264pay (pay0)
│
├── /hd    → videotestsrc(720p) → x264enc(ultrafast) → rtph264pay (pay0)
│
└── /file  → filesrc → decodebin → x264enc → rtph264pay (pay0)
                               └→ avenc_aac → rtpmp4apay (pay1)
```

## Konfigurasi Lanjutan

Edit `main.cpp` untuk menyesuaikan:
- **Bitrate**: parameter `bitrate=` pada `x264enc`
- **Resolusi**: parameter `width=` dan `height=`
- **Framerate**: parameter `framerate=`
- **Latency**: `gst_rtsp_media_factory_set_latency()`
- **Shared mode**: `gst_rtsp_media_factory_set_shared()` — TRUE=semua client share satu pipeline

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `x264enc` tidak ditemukan | Install `gstreamer1.0-plugins-ugly` |
| `avenc_aac` tidak ditemukan | Install `gstreamer1.0-libav` |
| Webcam tidak terdeteksi | Cek `/dev/video0`, ganti ke `videotestsrc` |
| Port sudah dipakai | Ganti port: `./rtsp_server 9000` |
