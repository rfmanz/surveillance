from ultralytics import YOLO
import cv2
import numpy as np
import telegram_send
import subprocess

src = "E:/sample.mp4"

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(src)

TARGET_CLASSES = [0, 2, 3, 5, 7]

object_tracker = {}
object_notified = set()
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
                    center_x = (b[0] + b[2]) // 2
                    center_y = (b[1] + b[3]) // 2

                    object_id = f"{cls}_{center_x}_{center_y}"
                    
                    if object_id in object_tracker:
                        object_tracker[object_id] += 1
                        if object_tracker[object_id] > 3 and object_id not in object_notified:
                            subprocess.run(["telegram-send", f"{CLASSES[cls]} detected"])
                            object_notified.add(object_id)
                    else:
                        object_tracker[object_id] = 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
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