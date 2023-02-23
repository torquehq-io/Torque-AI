import cv2
import torch
from flask import Flask, Response
from yolov5 import detect # assuming you have a `detect` function defined in a `yolov5.py` file

app = Flask(__name__)

# Load YOLOv5 model from .pt file
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/torquehqio/torquehq-io/main/Torque-AI/yolov5/yolov5s.pt')

# Define RTMP feed URLs and corresponding stream keys
FEEDS = {
    'feed1': {'url': 'rtmp://media5.ambicam.com:1938/live/1efa24f9-0cd0-47c5-b604-c7e3ee118302', 'key': 'stream1'},
    'feed2': {'url': 'rtmp://media5.ambicam.com:1938/live/3eb6f300-bb4a-4f66-af0c-15290833e3de', 'key': 'stream2'},
}

# Define YOLOv5 model configuration
CONFIG = {
    'img_size': 640,
    'conf_thres': 0.5,
    'iou_thres': 0.5,
}

# Define function to process RTMP feed frames and perform object detection
def process_feed(feed_url, stream_key):
    cap = cv2.VideoCapture(feed_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Perform object detection with YOLOv5 model
        detections = detect(model, frame, **CONFIG)
        # Send detections to output stream
        yield f"data: {detections}\n\n"

# Define Flask route to stream object detection results for each feed
@app.route('/detections')
def detections():
    def generate():
        for feed_name, feed_data in FEEDS.items():
            feed_url = feed_data['url']
            stream_key = feed_data['key']
            for data in process_feed(feed_url, stream_key):
                yield data
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
