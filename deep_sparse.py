import cv2
import subprocess
from deepsparse import Pipeline

src = "night.mp4"
model_path = "yolov8n.onnx"
yolo_pipeline = Pipeline.create(task='yolov8', model_path=model_path)

cap = cv2.VideoCapture(src)

TARGET_CLASSES = [0, 2, 3, 5, 7]

object_tracker = {}
object_notified = set()
CLASSES = {
    0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
}

def send_telegram_notification(class_name):
    try:
        subprocess.run(["telegram-send", f"{class_name} detected"], check=True)
        print(f"Notification sent for {class_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to send Telegram notification: {e}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Perform inference using DeepSparse
        results = yolo_pipeline(images=[frame])

        # Extract detections from the results
        boxes = results.boxes[0]
        scores = results.scores[0]
        labels = results.labels[0]

        detected_classes = set()

        for box, score, label in zip(boxes, scores, labels):
            cls = int(label)  # Convert to Python int
            conf = float(score)  # Convert to Python float
            
            if cls in TARGET_CLASSES and conf > 0.5:  # You can adjust the confidence threshold
                detected_classes.add(cls)

        # Update object tracker and send notifications
        for cls in detected_classes:
            if cls not in object_tracker:
                object_tracker[cls] = 1
            else:
                object_tracker[cls] += 1

            if object_tracker[cls] > 3 and cls not in object_notified:
                send_telegram_notification(CLASSES[cls])
                object_notified.add(cls)

        # Reset counters for objects not detected in this frame
        for cls in list(object_tracker.keys()):
            if cls not in detected_classes:
                del object_tracker[cls]

        # Display the frame (optional, for debugging)
        cv2.imshow('Live Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Type of results: {type(results)}")
    print(f"Structure of results: {results}")
finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("All windows closed.")