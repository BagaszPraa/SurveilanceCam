import cv2
import threading
import queue
import time
from lwcc import LWCC
import tempfile
import os
import numpy as np

class CrowdCounter:
    def __init__(self, video_source, model_name="CSRNet", model_weights="SHA", frame_interval=1.0):
        self.cap = cv2.VideoCapture(video_source)
        self.model_name = model_name
        self.model_weights = model_weights
        self.frame_interval = frame_interval  # detik antar hitung ulang

        self.latest_frame = None
        self.latest_count = 0
        self.latest_density = None
        self.lock = threading.Lock()
        self.running = True
        self.tmp_path = os.path.join(tempfile.gettempdir(), "crowd_frame.jpg")

    def _count_worker(self):
        while self.running:
            with self.lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is not None:
                cv2.imwrite(self.tmp_path, frame)
                count, density = LWCC.get_count(
                    self.tmp_path, model_name=self.model_name,
                    model_weights=self.model_weights, resize_img=False,
                    return_density=True
                )
                with self.lock:
                    self.latest_count = count
                    self.latest_density = density

            time.sleep(self.frame_interval)

    def run(self):
        worker = threading.Thread(target=self._count_worker, daemon=True)
        worker.start()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            with self.lock:
                self.latest_frame = frame
                count = self.latest_count
                density = self.latest_density

            display = frame.copy()
            if density is not None:
                h, w = frame.shape[:2]
                heatmap = cv2.resize(density, (w, h))
                heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                display = cv2.addWeighted(display, 0.6, heatmap_color, 0.4, 0)

            cv2.putText(display, f"Estimasi: {int(count)} orang", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Crowd Counter Realtime", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    counter = CrowdCounter("video/crowd_counting.mp4", frame_interval=1.0)
    counter.run()