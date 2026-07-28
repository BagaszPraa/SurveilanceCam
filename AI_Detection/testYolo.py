from ultralytics import YOLO
import cv2

model = YOLO("models/visDrone.pt")  # load a pretrained YOLOv8n model
TARGET_CLASSES = [0, 2, 3, 5, 7]
# 0 = person, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck, 1 = bicycle
VIDEO_PATH = "video/video_test.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, device=0, classes=[0], imgsz=832, conf=0.25, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("SurveilanceCam", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()