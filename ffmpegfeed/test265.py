import cv2
import subprocess as sp

# Get the default camera
cap = cv2.VideoCapture('rtmp://media5.ambicam.com:1938/live/1efa24f9-0cd0-47c5-b604-c7e3ee118302')

# Set the video frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Start FFmpeg process to push the RTMP stream
out = sp.Popen(['ffmpeg', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '25', '-i', '-', '-c:v', 'libx264', '-b:v', '1000k', '-c:a', 'aac', '-b:a', '128k', '-f', 'flv', 'rtmp://media.torqueai.io:1938/live/flask00'], stdin=sp.PIPE)

# Start capturing and pushing the video frames to FFmpeg
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.stdin.write(frame.tostring())

# Release the camera and terminate FFmpeg process
cap.release()
out.terminate()
