import cv2
import torch
from torchvision.transforms import functional as F

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/torqueai/github/main/Torque-AI/Users_slab/test/Models/yolov5s.pt')

# Set the threshold for object detection
confidence_threshold = 0.5

# Access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to the required format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection using the YOLOv5 model
    results = model(frame)

    # Get the detected objects
    objects = results.pred[0]

    for obj in objects:
        class_id = obj[5]
        confidence = obj[4]
        if confidence > confidence_threshold and class_id == 56:  # Assuming class ID 0 represents the specific object of interest
            bbox = obj[:4].int()

            # Extract bounding box coordinates
            x, y, x1, y1 = bbox

            # Calculate ROI coordinates by adding a certain distance (1 meter) to the bounding box
            distance = 1  # meters
            roi_x = max(0, int(x - distance * (x1 - x)))
            roi_y = max(0, int(y - distance * (y1 - y)))
            roi_x1 = min(frame.shape[1], int(x1 + distance * (x1 - x)))
            roi_y1 = min(frame.shape[0], int(y1 + distance * (y1 - y)))

            # Draw the bounding box and ROI zone on the frame
            cv2.rectangle(frame, (int(x), int(y)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x1, roi_y1), (0, 255, 255), 2)

    # Convert the frame back to BGR format for displaying
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
