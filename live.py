from ultralytics import YOLO
import cv2
import numpy as np

src = "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/Channels/101"

duration = 700
freq = 660

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(src)

TARGET_CLASSES = [0, 2, 3, 5, 7]

CLASSES = {
    0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in TARGET_CLASSES:
                    b = box.xyxy[0].cpu().numpy().astype(int)

                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    label = f"{CLASSES[cls]}"
                    cv2.putText(frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Live Detection', frame)

        key = cv2.waitKey(5)
        if key == 27:
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("All windows closed.")