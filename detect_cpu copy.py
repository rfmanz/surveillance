from ultralytics import YOLO
import cv2
import subprocess
import time

SHOW_VIDEO = True
TARGET_CLASSES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
STATIONARY_THRESHOLD = 30

# Load model with device='cpu' to force inference on CPU
model = YOLO('yolov8n.pt', device='cpu')
cap = cv2.VideoCapture("sample.mp4")

object_tracker = {}
last_notification_time = time.time()

if SHOW_VIDEO:
    window_name = 'Live Detection'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on a resized frame to reduce computational cost
        resized_frame = cv2.resize(frame, (640, 480))
        results = model(resized_frame, verbose=False, classes=list(TARGET_CLASSES.keys()))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                b = box.xyxy[0].cpu().numpy().astype(int)
                # Scale bounding box coordinates back to original frame size
                b = [int(x * frame.shape[1] / 640) if i % 2 == 0 else int(x * frame.shape[0] / 480) for i, x in enumerate(b)]
                center_x, center_y = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                object_id = f"{cls}_{center_x}_{center_y}"

                if object_id in object_tracker:
                    object_tracker[object_id] += 1
                    if object_tracker[object_id] > STATIONARY_THRESHOLD and time.time() - last_notification_time > 60:
                        subprocess.run(["telegram-send", f"Stationary {TARGET_CLASSES[cls]} detected"])
                        last_notification_time = time.time()
                        if SHOW_VIDEO:
                            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                else:
                    object_tracker[object_id] = 1

                if SHOW_VIDEO:
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    label = f"{TARGET_CLASSES[cls]}: {object_tracker[object_id]}"
                    cv2.putText(frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if SHOW_VIDEO:
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
