import cv2
import time
import torch
import feeds as fct
import subprocess as sp


def feed3():

    # to get the frame rate
    pTime = 0   # previous time
    cTime = 0   # current time

    video_cap = cv2.VideoCapture('rtmp://media5.ambicam.com:1938/live/c7047a1c-4de6-47bc-8cf9-6f5866875d2f') # number of the webcam, here the first
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))
    rtmp_url = "rtmp://media.torqueai.io:1938/live/flask3"
    ffmpeg = "ffmpeg -f rawvideo -pix_fmt bgr24 -s {}x{} -r 30 -i - -c:v libx264 -preset ultrafast -f flv {}".format(frame_width, frame_height, rtmp_url)

    process = sp.Popen(ffmpeg.split(), stdin=sp.PIPE)

    # Model
    #model_detection = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model_detection = torch.hub.load("ultralytics/yolov5", "custom", path = "./yolov5s.pt", force_reload=True)

    # Using the cam
    while True:
        # get the frame
        success, img = video_cap.read()

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

    

# if we are running this scipt,
if __name__ == "__main__":
    # then do this:
    feed3()
    
