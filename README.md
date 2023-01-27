# [Torque AI Auto Annotation Tool](http://torquehq.io)..
....
STEPS:
0. goto directory ( Torque-AI-Auto-Annotation-Tool )

1. create virtualenv:
- virtualenv venv

2. Activate venv:
- source venv/bin/activate

3. install flask dependencies:
- pip install -r requirements.txt
- pip install cmake
- pip install dlib

4. install yolov5 dependencies:
- cd yolov5
- pip install -r requirements.txt

5. Goto main dir Torque-AI-Auto-Annotation-Tool:
- cd ..

6. Execute python file (run.py):
- python3 run.py
- copy the link from terminal ( http://127.0.0.1:5000/ )
- paste in browser

7. ffmpeg with yolov5 
- cd yolov5/
- python3 d.py --source rtmp://media1.ambicam.com:1938/dvr7/fd9b67cc-6c2e-46c6-99c4-0e13ac403e32
Note: the frame will open in PC and ffmpeg will push the stream in NMS (http://media.torqueai.io:8080/admin/streams##)