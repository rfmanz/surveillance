# conda create -n surveillance python=3.9
# conda activate surveillance
# pip install ultralytics opencv-python numpy


from ultralytics import YOLO
import cv2
import numpy as np

src = "rtsp://admin:Hik12345@192.168.1.64:554/Streaming/Channels/101"
# src = "E:/sample.mp4"

duration = 700  # milliseconds
freq = 660  # Hz

model = YOLO('yolov8n.pt')

# Initialize capture from the video stream
cap = cv2.VideoCapture(src)

# Define the region of interest (ROI) - adjust these values as needed
roi_x, roi_y, roi_w, roi_h = 300, 300, 400, 300

# Classes we're interested in (person and vehicle)
TARGET_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

# Dictionary to store object positions and frame counts
# object_tracker = {}
CLASSES = {
    0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
}

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Draw ROI on frame
        # cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Predict using the full frame
        results = model(frame, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in TARGET_CLASSES:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    # center_x = (b[0] + b[2]) // 2
                    # center_y = (b[1] + b[3]) // 2

                    # Check if object is in ROI
                    # if roi_x < center_x < roi_x + roi_w and roi_y < center_y < roi_y + roi_h:
                    # object_id = f"{cls}_{center_x}_{center_y}"
                    
                    # if object_id in object_tracker:
                    #     object_tracker[object_id] += 1
                    #     if object_tracker[object_id] > 3:  # Object stationary for 30 frames
                    #         cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    #         winsound.Beep(freq, duration)
                    # else:
                    #     object_tracker[object_id] = 1

                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                    label = f"{CLASSES[cls]}"
                    # label = f"{CLASSES[cls]}: {object_tracker[object_id]}"
                    cv2.putText(frame, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # winsound.Beep(freq, duration)
        # Display the frame with detections
        cv2.imshow('Live Detection', frame)

        # Wait for a key press for 1 millisecond
        key = cv2.waitKey(5)
        if key == 27:  # Exit on ESC
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if cap is not None:
        cap.release()  # Release the video capture object
    cv2.destroyAllWindows()
    print("All windows closed.")