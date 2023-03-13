import cv2
import numpy as np


# Load the pre-trained YOLOv3 model
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Define the classes to detect
classes = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Loop through the video frames
while True:
    # Read the frame from the video capture
    ret, frame = cap.read()

    # Detect objects in the frame using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in classes:
                center_x, center_y = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0])
                w, h = int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply image masking to hide everything except the detected objects
    mask = np.zeros_like(frame)
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        mask[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
    frame = mask

    # Display the resulting video stream
    cv2.imshow('Detected Objects', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
