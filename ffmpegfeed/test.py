import cv2
import threading
import torch

# Load YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# def detect_objects(frame):
#     # Run object detection on the frame
#     detections = model(frame)
#     # Draw the detections on the frame
#     for detection in detections:
#         x1, y1, x2, y2, confidence, class_name = detection
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open video capture

# Start the detection thread
detection_thread = threading.Thread(args=(cap,))
detection_thread.start()

# Show the video frame
while True:
    success, img = cap.read()

    if success:
        new_img = fct.draw_anchor_box(img, fct.detection(img, model_detection))

        # get the frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # display the fps on the screen
        cv2.putText(
            new_img,                    # draw on the screen
            str(int(fps)),          # round fps
            (10, 70),               # text position
            cv2.FONT_HERSHEY_PLAIN, # font type
            3,                      # fonct scale
            (255, 0, 255),          # color purple
            3                       # thickness
        )
        process.stdin.write(img.tobytes())
        

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
