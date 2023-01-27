# [Torque AI Auto Annotation Tool](http://torquehq.io)..
....
STEPS:
0. goto directory ( Torque-AI )

1. install yolov5 dependencies:
- cd yolov5
- pip install -r requirements.txt

2. ffmpeg with yolov5 
- cd yolov5/
- python3 d.py --source rtmp://media1.ambicam.com:1938/dvr7/fd9b67cc-6c2e-46c6-99c4-0e13ac403e32
Note: the frame will open in PC and ffmpeg will push the stream in NMS (http://media.torqueai.io:8080/admin/streams##)