from lwcc import LWCC
import cv2
import numpy as np
import tempfile
import os

VIDEO_PATH = "video/crowd_counting.mp4"
# OUTPUT_PATH = "output/hasil_crowd.mp4"
ALERT_THRESHOLD = 500  # ganti sesuai kebutuhan, alert kalau estimasi > angka ini

tmp_path = os.path.join(tempfile.gettempdir(), "lwcc_frame.jpg")

cap = cv2.VideoCapture(VIDEO_PATH)
# cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

os.makedirs("output", exist_ok=True)
# out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_skip = 5  # hitung tiap 5 frame (density model berat, nggak perlu tiap frame)
frame_idx = 0
last_count = 0
last_density_map = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_skip == 0:
        cv2.imwrite(tmp_path, frame)
        count, density_map = LWCC.get_count(
            tmp_path, model_name="CSRNet", model_weights="SHA",
            resize_img=False, return_density=True
        )
        last_count = count
        last_density_map = density_map

    # buat overlay heatmap di atas frame asli
    display_frame = frame.copy()
    if last_density_map is not None:
        heatmap = cv2.resize(last_density_map, (w, h))
        heatmap = (heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        display_frame = cv2.addWeighted(display_frame, 0.6, heatmap_color, 0.4, 0)

    # info jumlah orang
    color = (0, 0, 255) if last_count > ALERT_THRESHOLD else (0, 255, 0)
    cv2.putText(display_frame, f"Estimasi: {int(last_count)} orang", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if last_count > ALERT_THRESHOLD:
        cv2.putText(display_frame, "⚠ KERUMUNAN PADAT", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Crowd Density Monitor", display_frame)
    # out.write(display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
# out.release()
cv2.destroyAllWindows()
# print(f"Video hasil disimpan di: {OUTPUT_PATH}")