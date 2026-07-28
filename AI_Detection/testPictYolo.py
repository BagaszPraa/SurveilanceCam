from ultralytics import YOLO
import cv2

model = YOLO("models/visDrone.pt")  # load model VisDrone
TARGET_CLASSES = [0]  # sesuaikan dengan index class di model VisDrone kamu (cek model.names)

IMAGE_PATH = "video/crowd_counting.png"       # ganti sesuai nama file foto kamu
OUTPUT_PATH = "video/result.jpg"     # nama file hasil deteksi

frame = cv2.imread(IMAGE_PATH)

# results = model.predict(frame, device=0, classes=TARGET_CLASSES, imgsz=832, conf=0.25, verbose=False)
results = model.predict(frame, device=0, classes=TARGET_CLASSES, imgsz=832, conf=0.25, iou=0.4, verbose=False)
annotated = results[0].plot()

cv2.imwrite(OUTPUT_PATH, annotated)
print(f"Hasil deteksi disimpan di: {OUTPUT_PATH}")

cv2.imshow("Hasil Deteksi", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()