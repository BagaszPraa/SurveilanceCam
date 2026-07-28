from ultralytics import YOLO
import cv2

model = YOLO("models/yolo11s.engine")

# cek dulu semua class yang dikenali model ini
print(model.names)

# VIDEO_PATH = "video/crowd_counting.mp4"
# VIDEO_PATH = "/home/bagas/Videos/bike.mp4"
# cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(0)  # untuk webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # tanpa "classes=..." → deteksi SEMUA class yang model kenali
    results = model.predict(frame, device=0, conf=0.25, verbose=False)

    annotated = results[0].plot(labels=True, conf=False, line_width=1)

    cv2.imshow("SurveilanceCam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()