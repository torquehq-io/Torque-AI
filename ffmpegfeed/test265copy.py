import cv2
import numpy as np
import requests
cap = cv2.VideoCapture('rtmp://media5.ambicam.com:1938/live/1efa24f9-0cd0-47c5-b604-c7e3ee118302')
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'HEVC')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
out.write(frame)
url = 'rtmp://media.torqueai.io:1938/live/flask00'
files = {'video': ('output.mp4', open('output.mp4', 'rb'), 'video/mp4')}

requests.post(url, files=files)
