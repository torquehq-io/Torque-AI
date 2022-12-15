# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

import os
from   flask_migrate import Migrate
from   flask_minify  import Minify
from   sys import exit

from apps.config import config_dict
from apps import create_app, db

# WARNING: Don't run with debug turned on in production!
DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:

    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)
    
if DEBUG:
    app.logger.info('DEBUG            = ' + str(DEBUG)             )
    app.logger.info('FLASK_ENV        = ' + os.getenv('FLASK_ENV') )
    app.logger.info('Page Compression = ' + 'FALSE' if DEBUG else 'TRUE' )
    app.logger.info('DBMS             = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT      = ' + app_config.ASSETS_ROOT )
################################################################################
############# Auto Annotation tool ##################

from flask import Flask, render_template, request, json, session, Response, url_for
import os, base64, random
from datetime import timedelta, datetime
from os.path import join, dirname, realpath

import time
from datetime import date
import threading
from autoAnnotation.DlibTracker import DlibTracker
from pathlib import Path
from pascal_voc_writer import Writer
from PyQt5.QtGui import QImage
from autoAnnotation.pascal_voc_io import PascalVocWriter
from autoAnnotation.pascal_voc_io import PascalVocReader
from autoAnnotation.yolo_io import YoloReader
from autoAnnotation.yolo_io import YOLOWriter
import os.path
import sys
import subprocess
import torch
import glob
import cv2
#####################################################
roi_x = 0
roi_y = 0
roi_w = 0
rou_h = 0
gTracker = None
gLabel = ''
gPath = os.getcwd()

class VideoCamera():
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0

    def __del__(self):
        self.video.release()

    def reset(self):
        self.video.release()
        self.video = cv2.VideoCapture(self.url)
        self.error_count = 0

    def get_frame(self):
        global gTracker, gLabel
        success, image = self.video.read()
        if success:

            if gTracker is not None:

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                gTracker.update(rgb)
                track_pos = gTracker.getPos()
                x1 = round(track_pos.left())
                x2 = round(track_pos.right())
                y1 = round(track_pos.top())
                y2 = round(track_pos.bottom())
                width = (int(x2) - int(x1))
                height = (int(y2) - int(y1))

                if len(gLabel) > 0:  # save tracking result
                    writer = Writer(gPath + "/" + gLabel + "_" +
                                    str(gTracker.cnt)+".jpg", width, height)
                    writer.addObject(gLabel, int(
                        x1), int(y1), int(x2), int(y2))
                    writer.save(gPath + "/" + gLabel + "_" +
                                str(gTracker.cnt)+".xml")
                    cv2.imwrite(gPath + "/" + gLabel+"_" +
                                str(gTracker.cnt)+'.jpg', image)
                # draw tracking result
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, (160, 90)))
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes(), True
        else:
            return None, False


def gen(camera):
    global gTracker
    gTracker = None
    while True:
        try:
            frame, suc = camera.get_frame()
            if suc:
                camera.error_count = 0
            else:
                camera.error_count += 1
                if camera.error_count > 5:
                    camera.reset()
                    return
                elif camera.error_count > 50:
                    ret, jpeg = cv2.imencode('.jpg', cv2.imread(
                        'static/images/no connected.jpg'))
                    frame = jpeg.tobytes()
        except:
            camera.error_count += 1
            if camera.error_count > 5:
                camera.reset()
                return
            elif camera.error_count > 50:
                ret, jpeg = cv2.imencode('.jpg', cv2.imread(
                    'static/images/no connected.jpg'))
                frame = jpeg.tobytes()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')




############################   Camera Stream API Url  ############################


@app.route('/video_feed1')
def fr_video_feed2():
    url = request.args.get('url')
    return Response(gen(VideoCamera(url)), mimetype='multipart/x-mixed-replace; boundary=frame')



#### Set label and tracking region ########


@app.route('/api/addLabel', methods=['POST'])
def api_addLabel():
    print("---Add label---")
    x = request.form.get('x', type=int)
    y = request.form.get('y', type=int)
    w = request.form.get('w', type=int)
    h = request.form.get('h', type=int)
    label = request.form.get('label')
    global gTracker
    gTracker = DlibTracker()
    gTracker.reset(x, y, w, h)
    global gLabel, gPath
    gLabel = label
    gPath = str(os.getcwd())+"/" + gLabel
    print(gPath)
    Path(gPath).mkdir(parents=True, exist_ok=True)
    Path(gPath + "/Images")
    Path(gPath + "/Labels")
    
    global dataset_path
    dataset_path = (str(os.getcwd()) +'/'+ gLabel)

    print("Spliting", dataset_path)
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images/train").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/images/valid").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels/train").mkdir(parents=True, exist_ok=True)
    Path(dataset_path + "/data/labels/valid").mkdir(parents=True, exist_ok=True)
    with open(dataset_path + "/data/" + 'data.yaml', 'w') as data:
        data.write('train:'+' '+dataset_path + "/data/images/train" + '\n')
        data.write('val:'+' '+dataset_path + "/data/images/valid" + '\n')
        data.write('\n')
        data.write('nc: 1')
        data.write('\n')

        data.write('names: [' + "'" + gLabel + "'" ']')
        data.write('\n')
        # data.write('NC: 1')
        # data.write('\n')
        data.write('SAVE_VALID_PREDICTION_IMAGES: True')



    with open(gLabel + '/' + 'classes.txt', 'w') as data1:
        data1.write(gLabel)

    return json.dumps({
        'status': 200,
        'msg': 'ok'
    })


print(gPath)
imgFolderPath = (str(os.getcwd()))


@app.route('/convert')
def convertYolo():
    print("---Convert---")
    print(gLabel)
    print(imgFolderPath)
    for file in os.listdir(gLabel):

        print(gLabel)
        if file.endswith(".xml"):
            print(gPath)
            print("Convert", file)

            annotation_no_xml = os.path.splitext(file)[0]

            imagePath = os.path.join(
                imgFolderPath + "/" + gLabel, annotation_no_xml + ".jpg")

            print("Image path:", imagePath)

            image = QImage()
            image.load(imagePath)
            imageShape = [image.height(), image.width(),
                          1 if image.isGrayscale() else 3]
            imgFolderName = os.path.basename(imgFolderPath + "/" + gLabel)
            imgFileName = os.path.basename(imagePath)

            writer = YOLOWriter(imgFolderName, imgFileName,
                                imageShape, localImgPath=imagePath)

            # Read classes.txt
            classListPath = imgFolderPath + "/" + gLabel + "/" + "classes.txt"
            classesFile = open(classListPath, 'r')
            classes = classesFile.read().strip('\n').split('\n')
            classesFile.close()

            # Read VOC file
            filePath = imgFolderPath + "/" + gLabel + "/" + file
            tVocParseReader = PascalVocReader(filePath)
            shapes = tVocParseReader.getShapes()
            num_of_box = len(shapes)

            for i in range(num_of_box):
                label = classes.index(shapes[i][0])
                xmin = shapes[i][1][0][0]
                ymin = shapes[i][1][0][1]
                x_max = shapes[i][1][2][0]
                y_max = shapes[i][1][2][1]

                writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

            writer.save(targetFile=imgFolderPath + "/" +
                        gLabel + "/" + annotation_no_xml + ".txt")
    return "Nothing"


dataset_path = (str(os.getcwd()))
percentage_test = 20
p = percentage_test/100
def split():
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        #print("for", gLabel)
        if random.random() <= p:
            print("for", gPath)
            os.system(
                f"cp {dataset_path}/{title}.jpg {gLabel}/data/images/train")
            os.system(
                f"cp {dataset_path}/{title}.txt {gLabel}/data/labels/train")
        else:
            os.system(
                f"cp {dataset_path}/{title}.jpg {gLabel}/data/images/valid")
            os.system(
                f"cp {dataset_path}/{title}.txt {gLabel}/data/labels/valid")



@app.route('/trainingModel')
def training():
    split()

    data1 = dataset_path + "/data/" +'data.yaml'
    subprocess.run(['python3', 'yolov5/train.py', '--data', data1, '--name', gLabel])
     
    
    
    return "None"  

model_path = (str(os.getcwd()))    
@app.route('/downloadModel')
def download():
    Path = (model_path + "/yolov5/runs/train" + '/' + gLabel + '/weights/best.pt')
    print(Path)
    return send_file(Path, as_attachment=True) 

import io
import os
from PIL import Image
import numpy as np
loadModel = (str(os.getcwd()))


from io import BytesIO
@app.route('/detectModel')
def genDetect():
    path_model = loadModel + "/yolov5/train/" + gLabel + "/" + "weights" + "/" + "best.pt"
    print("model pARTHHH", path_model)

    #model = torch.hub.load("ultralytics/yolov5", "custom", path = path_model ,force_reload=True)
    model = torch.hub.load('yolov5', 'custom', path=loadModel + "/yolov5/runs/" + "/" + "train" + "/" + gLabel + "/" + "weights" + "/" + "best.pt", source='local', force_reload=True)
    # Set Model Settings
    model.eval()
    model.conf = 0.6  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1) 
    cap=cv2.VideoCapture('rtmp://media5.ambicam.com:1938/live/81f911d4-84b2-43a9-98bd-bdba4dc4538f')
    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = cap.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = model(img, size=640)
           
            results.print()  # print results to screen
            
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR

        else:
            break

        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        #print(frame)
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/videoDetect')
def videoD():
    """Video streaming route. Put this in the src attribute of an img tag."""

    return Response(genDetect(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
###################################################################################

#####################fr candidate addon####################
from flask import Flask, Response, json, render_template
from werkzeug.utils import secure_filename
from flask import request
from os import path, getcwd
import time
import os
import cv2
from mtcnn import MTCNN
import numpy as np


##############################
roi_x = 0
roi_y = 0
roi_w = 0
rou_h = 0
gTracker = None
gLabel = ''
gPath = os.getcwd()



class FR_candidate_addon(object):
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0

    def __del__(self):
        self.video.release()

    
    def get_frame(self):
        

        fn_haar = 'haarcascade_frontalface_default.xml'
        haar_cascade = cv2.CascadeClassifier(fn_haar)
        success, frame = self.video.read()
        print("hello")

        if success:
            count = 0
            size = 4
            # fn_haar = 'haarcascade_frontalface_default.xml'
            fn_dir = str(os.getcwd())+'/database/'
            fn_name = gLabel
            print(fn_name )
            path = os.path.join(fn_dir, fn_name)
            # if not os.path.isdir(path):
            #     os.mkdir(path)
            (im_width, im_height) = (112, 112)
            # haar_cascade = cv2.CascadeClassifier(fn_haar)
            
            frame= cv2.flip(frame, 1, 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mini = cv2.resize(gray,(gray.shape[1]//size, gray.shape[0]//size))
            faces = haar_cascade.detectMultiScale(mini)
            faces = sorted(faces, key=lambda x: x[3])
            if faces:
                face_i = faces[0]
                (x, y, w, h) = [v * size for v in face_i]
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                pin=sorted([int(n[:n.find('.')]) for n in os.listdir(path)
                    if n[0]!='.' ]+[0])[-1] + 1
                cv2.imwrite('%s/%s.png' % (path, pin), face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, fn_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,
                    1,(0, 255, 0))
                time.sleep(0.38)
                count += 1
            
            print(str(count) + " images taken and saved to " + fn_name +" folder in database ")

                    
               
               
            #ret, jpeg = cv2.imencode('.jpg', cv2.resize(image, (160, 90)))
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), True
        else:
            return None, False


def gen_frca(camera):
    while True:
        frame, suc = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed_frca')
def frc_video_feed():
    url = request.args.get('url')
    return Response(gen_frca(FR_candidate_addon(url)), mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/api/addLabelfrca', methods=['POST'])
def api_addLabelfrca():
    print("---Add label---")
    
    label = request.form.get('label')
    global gTracker
    gTracker = DlibTracker()
   
    global gLabel, gPath
    gLabel = label
    gPath = str(os.getcwd())+"/database/" + gLabel
    print(gPath)
    Path(gPath).mkdir(parents=True, exist_ok=True)
    Path(gPath + "/Images")
 


    return json.dumps({
        'status': 200,
        'msg': 'ok'
    })

###################frca model training##################

from FR.generate_face_embeddings import GenerateFaceEmbedding
from FR.facial_recognition_model_training import TrainFaceRecogModel



@app.route('/frca_modeltrain')
def frca_modeltrain():
    gfe = GenerateFaceEmbedding()
    gfe.genFaceEmbedding(gPath)
    frmt = TrainFaceRecogModel()
    frmt.trainKerasModelForFaceRecognition()

    return None

################## Face recognition ####################
from mtcnn import MTCNN
import warnings
import sys
import dlib

from keras.models import load_model
import numpy as np
import pickle
import cv2
from imutils.video import FPS
import os
from FR import face_model
##################
class FacePredictor():
    def __init__(self,url):
       

      

        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0
        
        self.frame_width = int(self.video.get(3))
        self.frame_height = int(self.video.get(4))
        print(str(self.frame_width) + " : " + str(self.frame_height))
        self.save_width = 800
        self.save_height = int(800 / self.frame_width * self.frame_height)

    


    def detectFace(self):
         
        
                
        size = 4
        haar_file = 'haarcascade_frontalface_default.xml'
        (width, height) = (128, 128)
        face_cascade = cv2.CascadeClassifier(haar_file)
      
        embeddings = os.path.sep.join(
            [str(os.getcwd()), "FR/faceEmbeddingModels/embeddings.pickle"])
        le = os.path.sep.join(
            [str(os.getcwd()),  "FR/faceEmbeddingModels/le.pickle"])

        # Load embeddings and labels
        data = pickle.loads(open(embeddings, "rb").read())
        le = pickle.loads(open(le, "rb").read())

        embeddings = np.array(data['embeddings'])
        labels = le.fit_transform(data['names'])

        # Load the classifier model
        model = load_model(os.path.sep.join(
            [str(os.getcwd()), "FR/faceEmbeddingModels/my_model.h5"]))
        success ,im = self.video.read()
        
        if success:
            im = cv2.resize(im, (self.save_width, self.save_height))
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                # Try to recognize the face
                
                prediction = model.predict(face_resize)
                j = np.argmax(prediction)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if prediction[1] < 500:
                    name =  le.classes_[j]
                    text = "{}".format(name)
                    cv2.putText(im,text, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

                else:
                    cv2.putText(im, 'not recognized', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            ret, jpeg = cv2.imencode('.jpg', im)
            return jpeg.tobytes(), True
        else:
            return None, False

           
            
            


def gen_fr(camera):
    while True:
        frame, suc = camera.detectFace()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/video_feed_fr')
def fr_video_feed():
    url = request.args.get('url')
    return Response(gen_fr(FacePredictor(url)), mimetype='multipart/x-mixed-replace; boundary=frame')





#################################################################################



if __name__ == "__main__":
    app.run()
