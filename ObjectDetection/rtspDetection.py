import threading
import cv2

from ultralytics import YOLO

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

# ==========================================================
# Konfigurasi
# ==========================================================
MODEL_PATH = "models/yolo11s.engine"
CAM_INDEX  = 0                     # index kamera USB, sesuaikan (0, 1, ...)
WIDTH      = 1280
HEIGHT     = 720
FPS        = 30
CONF_THRESH = 0.25

RTSP_PORT  = "8554"
RTSP_MOUNT = "/stream"

model = YOLO(MODEL_PATH)
print(model.names)  # cek semua class yang dikenali model ini

# ==========================================================
# Capture di thread terpisah, selalu simpan frame TERBARU
# (mencegah lag akibat frame lama menumpuk di buffer, lihat
# catatan sebelumnya soal buffering di versi C++)
# ==========================================================
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError(f"Gagal buka kamera index {CAM_INDEX}")

latest_frame = None
frame_lock = threading.Lock()
running = True


def capture_loop():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[Capture] Gagal ambil frame, stream berakhir.")
            running = False
            break
        with frame_lock:
            latest_frame = frame


capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()


# ==========================================================
# RTSP Media Factory: bangun pipeline output dan push frame
# tiap kali GStreamer minta data baru (need-data signal)
# ==========================================================
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super().__init__()
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=time "
            f"caps=video/x-raw,format=BGR,width={WIDTH},height={HEIGHT},framerate={FPS}/1 "
            "! videoconvert ! video/x-raw,format=I420 "
            "! x264enc speed-preset=ultrafast tune=zerolatency bitrate=2048 key-int-max=30 "
            "! rtph264pay config-interval=1 name=pay0 pt=96"
        )
        self.number_frames = 0
        self.duration = int(1 / FPS * Gst.SECOND)

    def on_need_data(self, src, length):
        global latest_frame

        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            return  # belum ada frame dari kamera, skip giliran ini

        # ---- Inferensi YOLO ----
        results = model.predict(frame, device=0, conf=CONF_THRESH, verbose=False)
        annotated = results[0].plot(labels=True, conf=False, line_width=1)

        data = annotated.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = timestamp
        self.number_frames += 1

        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print(f"[RTSP] push-buffer gagal: {retval}")

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)


class GstServer(GstRtspServer.RTSPServer):
    def __init__(self):
        super().__init__()
        self.set_service(RTSP_PORT)
        factory = SensorFactory()
        factory.set_shared(True)  # semua client nonton stream yang sama (bukan instance baru tiap connect)
        self.get_mount_points().add_factory(RTSP_MOUNT, factory)
        self.attach(None)


if __name__ == "__main__":
    server = GstServer()
    loop = GLib.MainLoop()

    print("==========================================")
    print(" SurveilanceCam RTSP - konfigurasi")
    print(f" Kamera     : index {CAM_INDEX}")
    print(f" Model      : {MODEL_PATH}")
    print(f" Resolusi   : {WIDTH}x{HEIGHT}")
    print(f" RTSP Out   : rtsp://0.0.0.0:{RTSP_PORT}{RTSP_MOUNT}")
    print("==========================================")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        capture_thread.join(timeout=2)
        cap.release()
        print("Program selesai.")