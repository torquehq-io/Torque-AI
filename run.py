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

from datetime import date


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
    # app.logger.info('DATA_ROOT      = ' + app_config.DATA_ROOT )
################################################################################
############# Auto Annotation tool ##################

from flask import Flask, render_template, request, json, session, Response, url_for, send_file, redirect,stream_with_context,escape
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
import shutil
import inspect

# subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed1.py'])
# subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed2.py'])

# def ffmpegfeedall():
#     subprocess.Popen(['python3','feed1.py'])
#     #subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed1.py'])
#     # subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed2.py'])
#     # subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed3.py'])
#     # subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed4.py'])
    
# ffmpegfeedall()

#####################################################
roi_x = 0
roi_y = 0
roi_w = 0
rou_h = 0
gTracker = None
gLabel = ''
gPath = os.getcwd()
from flask_login import (
    current_user,
    login_user,
    logout_user
)
global current_loggin_user


class VideoCamera():
    def __init__(self, url):
        # cityList=db.execute("SELECT * FROM User_camera_sources order by username ")
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


@app.route('/autoLabel.html' ,methods=["POST","GET"])
def autoLabel():
    
   
    return render_template('home/autoLabel.html', User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))

############################   Camera Stream API Url  ############################


@app.route('/video_feed1')
def fr_video_feed2():
    url = request.args.get('url')
    return Response(gen(VideoCamera(url)), mimetype='multipart/x-mixed-replace; boundary=frame')



#### Set label and tracking region ########


@app.route('/api/addLabel', methods=['POST'])
def api_addLabel():
    current_loggin_user = current_user.username
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
    gPath = str(os.getcwd())+"/Users_slab/"+current_loggin_user+"/"+gLabel
    print(gPath)
    Path(gPath).mkdir(parents=True, exist_ok=True)
    Path(gPath + "/Images")
    Path(gPath + "/Labels")
    
    global dataset_path
    dataset_path = (str(os.getcwd()) +'/'+ gLabel)

    print("Spliting", gPath)
    Path(gPath).mkdir(parents=True, exist_ok=True)
    Path(gPath + "/model").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/images").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/labels").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/images/train").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/images/valid").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/labels/train").mkdir(parents=True, exist_ok=True)
    Path(gPath + "/data/labels/valid").mkdir(parents=True, exist_ok=True)
    with open(gPath + "/data/" + 'data.yaml', 'w') as data:
        data.write('train:'+' '+gPath + "/data/images/train" + '\n')
        data.write('val:'+' '+gPath + "/data/images/valid" + '\n')
        data.write('\n')
        data.write('nc: 1')
        data.write('\n')

        data.write('names: [' + "'" + gLabel + "'" ']')
        data.write('\n')
        # data.write('NC: 1')
        # data.write('\n')
        data.write('SAVE_VALID_PREDICTION_IMAGES: True')



    with open(gPath + '/' + 'classes.txt', 'w') as data1:
        data1.write(gLabel)

    return json.dumps({
        'status': 200,
        'msg': 'ok'
    })


print(gPath)



@app.route('/convert')
def convertYolo():
    current_loggin_user = current_user.username
    imgFolderPath = str(os.getcwd())+"/Users_slab/"+current_loggin_user
    print("---Convert---")
    print(gLabel)
    print(imgFolderPath)
    for file in os.listdir(imgFolderPath+"/"+gLabel):

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



percentage_test = 20
p = percentage_test/100
def split():
    current_loggin_user = current_user.username

    dataset_path = str(os.getcwd())+"/Users_slab/"+current_loggin_user+"/"+gLabel
    for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        #print("for", gLabel)
        if random.random() <= p:
            print("for", gPath)
            os.system(
                f"cp {dataset_path}/{title}.jpg {dataset_path}/data/images/train")
            os.system(
                f"cp {dataset_path}/{title}.txt {dataset_path}/data/labels/train")
        else:
            os.system(
                f"cp {dataset_path}/{title}.jpg {dataset_path}/data/images/valid")
            os.system(
                f"cp {dataset_path}/{title}.txt {dataset_path}/data/labels/valid")

def move_modelfile():
    global current_loggin_user,target_dir
    # user = Users.query.filter_by(username=LoginForm.username).first()
    if current_user.is_authenticated:
        current_loggin_userid = current_user.get_id() # return username in get_id()
        current_loggin_user = current_user.username
        print(current_loggin_userid)
        print(current_loggin_user )
    #custom_models = os.path.join(str(getcwd()+"/Users_slab/"+current_loggin_user+"/Models/"),gLabel)

    #os.makedirs(custom_models)
    source_dir = str(getcwd()+"/yolov5/runs/train/"+gLabel+"/")
    target_dir = str(getcwd()+"/Users_slab/"+current_loggin_user+'/'+gLabel+'/model')
    file_names = os.listdir(source_dir)

    
    

    #shutil.move(os.path.join(source_dir), target_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    print("Model file is successfully move at user folder")

    source_dir1 = str(getcwd()+"/Users_slab/"+current_loggin_user+'/'+gLabel+'/model/weights/'+gLabel+'.pt')
    target_dir1 = str(getcwd()+"/Users_slab/"+current_loggin_user+'/Models/')
    shutil.move(source_dir1, target_dir1)
    
    

    today = date.today()
    genrated_on= today.strftime("%B %d, %Y")

    add_model =User_Models_record(username=current_loggin_user,user_id=current_loggin_userid, model_name=gLabel,generated_date=genrated_on)
    db.session.add(add_model)

     
    db.session.commit()
        
    print("Model is stored in database successfully !!")

def rename_modelfile():
    source_filepath = (str(os.getcwd())+"/yolov5/runs/train/"+gLabel+"/weights/best.pt")
    rename_filepath =(str(os.getcwd())+"/yolov5/runs/train/"+gLabel+"/weights/"+gLabel+".pt")
    renamed_file=os.rename(source_filepath,rename_filepath)
    
    
    print("Successfully rename the model file name")
    print("model genrated on  ")
    move_modelfile()
    return None

@app.route('/trainingModel')
def training():
    split()
    current_loggin_user = current_user.username

    dataset_path = str(os.getcwd())+"/Users_slab/"+current_loggin_user+"/"+gLabel
    data1 = dataset_path + "/data/" +'data.yaml'
    subprocess.run(['python3','yolov5/train.py','--data', data1, '--name', gLabel])
    #subprocess.run(['python3','-m','torch.distributed.run','--nproc_per_node','2','yolov5/train.py','--data', data1, '--name', gLabel,'--device','0,1'])
    rename_modelfile()
       
    return "None"  

model_path = (str(os.getcwd()))    
@app.route('/downloadModel')
def download():
    Path = (model_path+"/Users_slab/"+current_loggin_user+"/Models/"+gLabel+'.pt')




    
    print(Path)
    return send_file(Path, as_attachment=True) 

import io
import os
from PIL import Image
import numpy as np


loadModel = (str(os.getcwd()))


from io import BytesIO



class Objdetection():
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0

        # self.model = torch.hub.load('yolov5', 'custom', path=loadModel + "/yolov5/runs/" + "/" + "train" + "/" + gLabel + "/" + "weights" + "/" + gLabel+".pt", source='local', force_reload=True)
        self.model = torch.hub.load('yolov5', 'custom', path=loadModel + '/yolov5/runs/train/'+gLabel+'/weights/'+gLabel+'.pt', source='local', force_reload=True)
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        path_model = loadModel + "/yolov5/train/" + gLabel + "/" + "weights" + "/" + gLabel+'.pt'
        print("model pARTHHH", path_model)

        # model = torch.hub.load("ultralytics/yolov5", "custom", path = "/home/torque/Desktop/Torque-AI/Rampage/Intact-core/Rampage_AI/yolov5/runs/train/2/weights/best.pt",force_reload=True)
        
        # model = torch.hub.load('yolov5', 'custom', path=loadModel + "/yolov5/runs/" + "/" + "train" + "/" + gLabel + "/" + "weights" + "/" + "best.pt", source='local', force_reload=True)
        
        
        # model = torch.hub.load('yolov5', 'custom', path='/home/torque/Desktop/Torque-AI/Rampage/Intact-core/Rampage_AI/yolov5/runs/train/a12/weights/a12.pt', source='local', force_reload=True)
        # Set Model Settings
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
    
        
            # Capture frame-by-fram ## read the camera frame
        success, frame = self.video.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results =self.model(img, size=640)
        
            results.print()  # print results to screen
            
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
       

def gen_det(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_det')
def det_video_feed():
    url = request.args.get('url')
    return Response(gen_det(Objdetection(url)), mimetype='multipart/x-mixed-replace; boundary=frame')


#######################################################################################################################
############################### load model ############################################################################
@app.route("/load_model")
def load_model():
   
    return render_template("home/loadmodel.html", User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username),User_Models_record=User_Models_record.query.filter_by(username=current_user.username))
class Objdetection_with_load_model():
    def __init__(self, url,model):
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.modelnme = model
        
        self.error_count = 0
        current_loggin_user = current_user.username
        if self.modelnme != "yolo5s":
        # self.model = torch.hub.load('yolov5', 'custom', path=loadModel + "/yolov5/runs/" + "/" + "train" + "/" + gLabel + "/" + "weights" + "/" + gLabel+".pt", source='local', force_reload=True)
            self.model = torch.hub.load('yolov5', 'custom', path=loadModel +'/Users_slab/' +current_loggin_user+'/Models/'+self.modelnme+'.pt', source='local', force_reload=True, device='cpu')
        else :
            self.model = torch.hub.load('yolov5', 'custom', path=loadModel+"/yolov5/yolov5s.pt" , source='local', force_reload=True, device='cpu')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global Row

        # model = torch.hub.load("ultralytics/yolov5", "custom", path = "/home/torque/Desktop/Torque-AI/Rampage/Intact-core/Rampage_AI/yolov5/runs/train/2/weights/best.pt",force_reload=True)

        # model = torch.hub.load('yolov5', 'custom', path=loadModel + "/yolov5/runs/" + "/" + "train" + "/" + gLabel + "/" + "weights" + "/" + "best.pt", source='local', force_reload=True)

        # model = torch.hub.load('yolov5', 'custom', path='/home/torque/Desktop/Torque-AI/Rampage/Intact-core/Rampage_AI/yolov5/runs/train/a12/weights/a12.pt', source='local', force_reload=True)
        # Set Model Settings
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1)

        # Capture frame-by-fram ## read the camera frame
        success, frame = self.video.read()
        if success == True:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = self.model(img, size=640)

            results.print()  # print results to screen

            # convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render())  # RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # BGR
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame


def gen_det_with_load_model(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_det_with_loadmodel')
def det_loadmodel_video_feed():
    url = request.args.get('url')
    
    return Response(gen_det_with_load_model(Objdetection_with_load_model(url,select_model)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_det_with_loadmodel_fetch')
def det_loadmodel_video_feed_fetch():
    global select_model
    select_model=request.args.get('model')
    print(select_model)
    return select_model




#################################################################################################################################
@app.route("/BWphotos")
def bwPhotos():
    binderList = os.listdir("/home/torquehq/torquehq-io/Github/Torque-AI/Users_slab/test/a1")
    binderList = ['test/a1/' + image for image in binderList]
    return render_template("images.html", binderList=binderList)
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
        self.video = cv2.VideoCapture(0)
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
            gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
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

    return 'Train completed'

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
       

      

        self.video = cv2.VideoCapture(0)
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

############### Person Counter #############################################
######import required libraries ####################
import cv2
import datetime
import imutils
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import os
from imutils.video import FPS
from person_counter.nms import non_max_suppression_fast
##############3import pretrained models###############
prototxtPath = os.path.sep.join(
    [str(os.getcwd()), "face_detector/customai.prototxt"])
weightsPath = os.path.sep.join(
    [str(os.getcwd()), "face_detector/customai.caffemodel"])

detector = cv2.dnn.readNetFromCaffe(
    prototxt=prototxtPath, caffeModel=weightsPath)



#######################Main code of person Counter######################

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]



import cv2
import threading
 
class RecordingThread (threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True
 
        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))
 
    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)
 
        self.out.release()
 
    def stop(self):
        self.isRunning = False
 
    def __del__(self):
        self.out.release()



class CentroidTracker:

    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries          <li class="nav-item"><a class="nav-link" href="autoannotation.html">
             
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            # return self.objects
            return self.bbox

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # return the set of trackable objects
        # return self.objects
        return self.bbox



 
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)




class Person_Counter(object):
    def __init__(self,url):
        self.vs = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0

        self.is_record = False
        self.out = None
 
        # Thread for recording
        self.recordingThread = None
       
        
   

    def __del__(self):
        cv2.destroyAllWindows()

    def non_max_suppression_fast(boxes, overlapThresh):
        try:
            if len(boxes) == 0:
                return

            if boxes.dtype.kind == "i":
                boxes = boxes.astype("float")

            pick = []

            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            idxs = np.argsort(y2)

            while len(idxs) > 0:
                last = len(idxs) - 1
                i = idxs[last]
                pick.append(i)

                xx1 = np.maximum(x1[i], x1[idxs[:last]])
                yy1 = np.maximum(y1[i], y1[idxs[:last]])
                xx2 = np.minimum(x2[i], x2[idxs[:last]])
                yy2 = np.minimum(y2[i], y2[idxs[:last]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)

                overlap = (w * h) / area[idxs[:last]]

                idxs = np.delete(idxs, np.concatenate(([last],
                                                       np.where(overlap > overlapThresh)[0])))

            return boxes[pick].astype("int")
        except Exception as e:
            print("Exception occurred in non_max_suppression : {}".format(e))


    def get_frame(self):
        self.fps_start_time = datetime.datetime.now()
        self.fps = 0
        self.total_frames = 0
        ret, frame = self.vs.read()
        frame = imutils.resize(frame, width=600)
        self.total_frames = self.total_frames + 1

        (H, W) = frame.shape[:2]
       
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0,
                                               i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 140), 2)
            text = "Counter: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (140, 0, 0), 1)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - self.fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (self.total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
  
        #self.fps.update()

        ret, jpeg = cv2.imencode('.jpg', frame)
    
        return jpeg.tobytes()

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.vs)
        self.recordingThread.start()
 
    def stop_record(self):
        self.is_record = False
 
        if self.recordingThread != None:
            self.recordingThread.stop()

#####################
def gen_pc(camera):
    while True:
        frame= camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
@app.route('/person_counter.html' ,methods=["POST","GET"])
def person_counter():
    
   
    return render_template('home/person_counter.html', User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))



@app.route('/video_feed_pc')
def pc_video_feed():
    url = request.args.get('url')
    return Response(gen_pc(Person_Counter(url)), mimetype='multipart/x-mixed-replace; boundary=frame')



########################## Fire detection ##########################

class Fire_detection():
    def __init__(self, url):
        # self.video = cv2.VideoCapture('rtmp://media1.ambicam.com:1938/dvr7/fd9b67cc-6c2e-46c6-99c4-0e13ac403e32')
        current_loggin_user=current_user.username
        fetch_url =  User_camera_sources.query.filter_by(username=current_loggin_user).first()
        print(fetch_url.link1)
        self.video = cv2.VideoCapture(fetch_url.link1)

        self.url = url
        self.error_count = 0
        self.model = torch.hub.load('yolov5', 'custom', path='Fire_detection/fire.pt', source='local', force_reload=True)

       

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
       

        
        # Set Model Settings Dynamic
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
        # Set Model Settings Static
        # model.eval()
        # model.conf = 0.6  # confidence threshold (0-1)
        # model.iou = 0.45  # NMS IoU threshold (0-1) 
        
            # Capture frame-by-fram ## read the camera frame
        
        success, frame = self.video.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = self.model(img, size=640)
        
            results.print()  # print results to screen
            
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
       

def gen_fire_det(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_fire_det')
def fire_det_video_feed():
    url = request.args.get('url')
    
    return Response(gen_fire_det(Fire_detection(url)), mimetype='multipart/x-mixed-replace; boundary=frame')






#################################################################################
class Fire_detection1():
    def __init__(self, url1):
        # self.video = cv2.VideoCapture('rtmp://media1.ambicam.com:1938/dvr7/3803ff24-f7cc-48bb-bc25-bbc5486ef728')
        current_loggin_user=current_user.username
        fetch_url =  User_camera_sources.query.filter_by(username=current_loggin_user).first()
        print(fetch_url.link2)
        self.video = cv2.VideoCapture(fetch_url.link2)

        self.url1 = url1
        self.error_count = 0
       
        self.model = torch.hub.load('yolov5', 'custom', path='/home/torqueai/github/main/Torque-AI/yolov5s.pt', source='local', force_reload=True)

       

    def __del__(self):
        self.video.release()
    
    def get_frame1(self):
       

        
       # Set Model Settings Dynamic
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
        # Set Model Settings Static
        # model.eval()
        # model.conf = 0.6  # confidence threshold (0-1)
        # model.iou = 0.45  # NMS IoU threshold (0-1) 
        
            # Capture frame-by-fram ## read the camera frame
        
        success, frame = self.video.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))
            results = self.model(img, size=640)
        
            results.print()  # print results to screen
            
            
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
       

def gen_fire_det1(camera1):
    while True:
        frame = camera1.get_frame1()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_fire_det1')
def fire_det_video_feed1():
    url1 = request.args.get('url1')
    
    return Response(gen_fire_det1(Fire_detection1(url1)), mimetype='multipart/x-mixed-replace; boundary=frame')

############### store the camera sources in database tabel ###################
from apps.authentication.models import Users


class User_camera_sources(db.Model):
    __tablename__ = 'User_camera_sources'
    id = db.Column('source_id', db.Integer, primary_key = True)
    username = db.Column(db.String(1000),unique=True)
    link1 = db.Column(db.String(3000))  
    link2 = db.Column(db.String(3000))
    link3= db.Column(db.String(3000))
    link4= db.Column(db.String(3000))

    def __init__(self, username,link1, link2,link3,link4):
        self.username = username
        self.link1 = link1
        self.link2 = link2
        self.link3 = link3
        self.link4 = link4
@app.route('/new', methods = ['GET', 'POST'])
def new():
    if request.method == 'POST':
    #   if not request.form['link1'] or not request.form['link2'] or not request.form['link3']:
    #      print('Please enter all the fields', 'error')
    #   else:
      if current_user.is_authenticated:
       
        current_loggin_user = current_user.username
        user = Users.query.filter_by(username=current_loggin_user).first()
        user=str(user)[21:-1]
        print(user)

        link1=request.form['link1']
        link2=request.form['link2']
        link3=request.form['link3']
        link4=request.form['link4']
        add_camerasources =User_camera_sources(username=current_loggin_user,link1=link1, link2=link2,
                link3=link3, link4=link4)
            
        db.session.add(add_camerasources)
        db.session.commit()
        
        print('Record was successfully added')
    return render_template('home/new.html')

###############################################################

@app.route('/storeRtmpLink')
def Test():
    current_loggin_user = current_user.username
    user = Users.query.filter_by(username=current_loggin_user).first()
    user=str(user)[21:-1]
    if user:
        db.session.execute("SELECT * FROM User_camera_sources")
        
        db.session.close()

    return render_template('home/camera_source_db.html',User_camera_sources=User_camera_sources.query.filter_by(username=current_user.username),data=current_user.username,table=User_camera_sources)


@app.route('/insert', methods = ['POST'])
def insert():
    if request.method == "POST":
        print("Data Inserted Successfully")
        link1= request.form['link1']
        link2 = request.form['link2']
        link3= request.form['link3']
        link4= request.form['link4']
        # cur = db.session.connection()
        current_loggin_user = current_user.username
        # db.session.execute("INSERT INTO User_camera_sources (link1,link2,link3,link4) VALUES (%s, %s, %s, %s,)," ,(link1, link2 , link3 , link4))
        add_camerasources =User_camera_sources(username=current_loggin_user,link1=link1, link2=link2,
                link3=link3, link4=link4)
       
        db.session.add(add_camerasources)

     
        db.session.commit()    
        
        return redirect(url_for('Test'))

# @app.route('/delete/<string:id_data>', methods = ['GET'])
# def delete(id_data):
#     print("Record Has Been Deleted Successfully")
   
#     db.session.execute("DELETE FROM User_camera_sources WHERE id=%s", (id_data,))
#     db.session.commit()
#     return redirect(url_for('Test'))



@app.route('/update', methods= ['POST', 'GET'])
def update():
    if request.method == 'POST':
        
       
        link1= request.form['link1']
        link2 = request.form['link2']
        link3= request.form['link3']
        link4= request.form['link4']
        current_loggin_user = current_user.username
        admin = User_camera_sources.query.filter_by(username=current_loggin_user).first()
        admin.link1 = link1
        admin.link2 = link2
        admin.link3 = link3
        admin.link4 = link4
        
        db.session.commit()
       
        print("Data Updated Successfully")
    return redirect(url_for('Test'))
########################################################

#################################################################################




@app.route('/test')
def test():
    # url = "/home/torquehq/torquehq-io/Github/Torque-AI/Users_slab/test/a1"
    # full_filename = os.path.join(app.config['img'], 'full.jpg')
    # return render_template("home/images.html",image = full_filename)

    url = "/home/torquehq/torquehq-io/Github/Torque-AI/apps/static/dgsb"
    imageList = os.listdir(url)
    imagelist = ['dgsb/' + image for image in imageList]
    return render_template("home/images.html", imagelist=imagelist)

############################################################################
##################  model display #################################
@app.route('/model_display')
def model_display():
 
    
    return render_template('home/model_display.html',User_Models_record=User_Models_record.query.filter_by(username=current_user.username))

@app.route('/delete/<int:id>', methods = ['GET','POST'])
def delete(id):
    print(id)
    current_loggin_user = current_user.username
    model_to_delete = User_Models_record.query.get_or_404(id)
    model_to_delete_file = User_Models_record.query.filter_by(serial_no=id).first()
    model_to_delete_local= model_to_delete_file.model_name
    print(model_to_delete)
    os.remove(os.path.join(str(os.getcwd()+"/Users_slab/"+current_loggin_user+"/Models/"), model_to_delete_local+".pt"))
    db.session.delete(model_to_delete)
  
   
    
    db.session.commit()
    print("Record Has Been Deleted Successfully")
    return redirect(url_for('model_display'))




@app.route('/rename_modelname/<int:id>', methods= ['POST', 'GET'])
def rename_modelname(id):
   
    model_to_rename = User_Models_record.query.get_or_404(id)
    if request.method == 'POST':
        current_loggin_user = current_user.username
        
        new_model_name= request.form['model_name']
        rename_to_model_filename_path =(os.path.join(str(os.getcwd()+"/Users_slab/"+current_loggin_user+"/Models/"), model_to_rename.model_name +".pt"))
        renamed_model_path=(os.path.join(str(os.getcwd()+"/Users_slab/"+current_loggin_user+"/Models/"), new_model_name +".pt"))
        renamed_file=os.rename(rename_to_model_filename_path,renamed_model_path)
        
        model_to_rename.model_name =new_model_name
        try:
           
            
            db.session.commit()
            # return render_template("model_display.html",form=form,model_to_rename = model_to_rename)
        except:
            print("model not rename...")
            


       
        print("model name Updated Successfully")
    return redirect(url_for('model_display'))


##############################################################################
################################### model storing at database ###################
class User_Models_record(db.Model):
    __tablename__ = 'Custom_Models_record'
    serial_no = db.Column(db.Integer, primary_key = True)
    user_id = db.Column('user_id', db.Integer)
    username = db.Column(db.String(1000))
    model_name = db.Column(db.String(3000))  
    generated_date = db.Column( db.Integer)

    def __init__(self,user_id  ,username,model_name,generated_date):

        self.user_id  = user_id 
        self.username = username
        self.model_name= model_name
        self.generated_date= generated_date




#####################################################################################
##################################camera sources database table ########################
class User_camera_sources_record(db.Model):
    __tablename__ = 'User_camera_sources_record'
    serial_no = db.Column(db.Integer, primary_key = True)
    user_id = db.Column('user_id', db.Integer)
    username = db.Column(db.String(1000))
    source = db.Column(db.String(3000))  
    name_source = db.Column(db.String(3000))

    def __init__(self,user_id  ,username,source,name_source):

        self.user_id  = user_id 
        self.username = username
        self.source= source
        self.name_source= name_source
######################################### uplode the camera source text file from user ###############
from distutils.log import debug
from fileinput import filename
import csv  


@app.route('/camera_src_display')
def camera_src_display():
    return render_template('home/camera_source_textfile.html',User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username),data=current_user.username,table=User_camera_sources)
@app.route('/camera_source_textfile', methods = ['POST'])  
def camera_source_textfile():  
    if request.method == 'POST':  
        f = request.files['file']
        f.save(f.filename)
        current_loggin_userid = current_user.get_id() # return username in get_id()
        current_loggin_user = current_user.username 
        with open (f.filename ,'r') as csv_file:
            csvfile= csv.reader(csv_file,delimiter=",")
            listformate_csvfile=list(csvfile)
            for row in listformate_csvfile:
                    if(len(row)!=0):
                       
                        source=row[0]

                       
                        name_source=row[1]
                        add_model =User_camera_sources_record(username=current_loggin_user,user_id=current_loggin_userid, source=source,name_source=name_source)
                        db.session.add(add_model)

        
                        db.session.commit()
                
                        print("SOURCE :  {} , CAMERA_NAME: {} is successfully added in database ! ".format(source,name_source))



           
        
        # print("Model is stored in database successfully !!")
            
        return render_template("home/camera_source_textfile.html", name = f.filename , User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username),data=current_user.username,table=User_camera_sources )  
@app.route('/insert_src', methods = ['POST'])
def insert_src():
    if request.method == "POST":
        print("Data Inserted Successfully")
        source_link = request.form['rtmpUrl']
        name_of_camera  = request.form['camera_name']
        current_loggin_userid = current_user.get_id() 
        current_loggin_user = current_user.username 
        add_model =User_camera_sources_record(username=current_loggin_user,user_id=current_loggin_userid, source=source_link,name_source=name_of_camera)
        db.session.add(add_model)
        db.session.commit()
        
        return redirect(url_for('camera_src_display'))
    
##### ---------------------- delete camera source -------------------------

@app.route('/delete_src/<int:id>', methods = ['GET','POST'])
def delete_src(id):
    print(id)
    current_loggin_user = current_user.username
    source_to_delete = User_camera_sources_record.query.get_or_404(id)
    source_to_delete_file = User_camera_sources_record.query.filter_by(serial_no=id).first()
    # source_to_delete_local= source_to_delete_file.model_name
    print(source_to_delete)
    #os.remove(os.path.join(str(os.getcwd()+"/Users_slab/"+current_loggin_user+"/Models/"), model_to_delete_local+".pt"))
    db.session.delete(source_to_delete)
  
   
    
    db.session.commit()
    print("Record Has Been Deleted Successfully")
    return redirect(url_for('camera_src_display'))

###########################################################################
object_list={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
              21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 
                61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
Row=[]
object, object2 = '', ''
class Objdetection_try():
   
  
    def __init__(self, url):
      
        print("in detect...................................")
        self.video = cv2.VideoCapture("rtmp://media5.ambicam.com:1938/live/1efa24f9-0cd0-47c5-b604-c7e3ee118302")
        self.url = url
        self.error_count = 0
        self.model = torch.hub.load('yolov5', 'custom', path='/home/torque/Desktop/main/torqueai/yolov5s.pt', source='local', force_reload=True)
       
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        global Row
        global object
        global object2
        
        
        # Set Model Settings
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
    
        
            # Capture frame-by-fram ## read the camera frame
        success, frame = self.video.read()
        if success == True:

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            
            #print(type(frame))

            img = Image.open(io.BytesIO(frame))

            # object =self.model(img, size=640)
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            keys=[0,1,2]
            values = list( map(names.get, keys) )
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT",values)
            pred = self.model(img, augment='store_true')
            # print("ttttttttt",pred.xyxy[0])  # im predictions (tensor)
            # print("pppnnnn",pred.pandas().xyxy[0])

            # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa",dir(pred.print()))
            # print("ggggggggggggggggggg",getattr(pred, 'pred.print()', 0))
            
            
            # object.print()  # print results to screen
          
            
            img = np.squeeze(pred.render())
              # batch_size >= 1
                # p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            p, object, pred,frame = path,'', pred,frame.count
                # print('p {} , s {} , img {},frame {} '.format(p , object,img,frame))
            det = pred.xyxy[0]
            # if len(pred.xyxy[0]):
                # Rescale boxes from img_size to im0 size
               
            # object+= '%gx%g ' % img.shape[2:]
            for c in  det[:, -1].unique():  # detections per class
                    n = (det[:, -1] == c).sum()
                    # object +=   str(names[int(c)])  # add to string
                    print("ccccccc:", c)
                    print("nnnnnnn:", n)
                    print("sssssss:", names[int(c)])
                    object += f"{n} {names[int(c)]}{'object' * (n > 1)}, "
                    print(object)

            
            if object == '':
                object = 'Empty.'
                print(object)
            else:
                object = object[:-2] + '.'
                print(object)    
            #convert remove single-dimensional entries from the shape of an array
            # img = np.squeeze(object.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            return frame
           

def gen_det_obj(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_det_try')
def det_video_feed_try():
    url = request.args.get('url')
    return Response(gen_det_obj(Objdetection_try(url)), mimetype='multipart/x-mixed-replace; boundary=frame')
    

def out():
    global object2
    global object
    while(True):
        if object2!=object and object.endswith("."):
           object2 = object
           yield "|--" + object2 + "----|" + "<br>"
        else:
            yield ''
           


def stream_template(template_name, **context):
    app.update_template_context(context)
    t = app.jinja_env.get_template(template_name)
    rv = t.stream(context)
    rv.disable_buffering()
    return rv



@app.route('/try', methods = ['POST', 'GET'])
def stream_view():
    global Row
    rows = []
    cknames = request.form.getlist('skills')
    print(cknames)
    for i in range(10):
      rows.append(str(i))
      print("you are in try {}".format(i))
    # rows = Row
    return Response(stream_with_context(stream_template('home/try.html', rows=Row,data=object_list,len = len(object_list))))
@app.route('/output')
def output():
    print("you are in output")
    return Response(out(), mimetype='text/html')

########################################################################3



#########################################################################
########## multi object detection list ###################3



class Objdetection_4multi():
   
  
    def __init__(self, url):
      
        print("in detect...................................")
        self.video = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0
        self.model = torch.hub.load('yolov5', 'custom', path='/home/torque/Desktop/main/torqueai/yolov5s.pt', source='local', force_reload=True)
        self.cknames = cknames
        
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        global Row
        
        
        
        
        # Set Model Settings
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
    
        
        # Capture frame-by-fram ## read the camera frame
        success, frame = self.video.read()
        if success == True:

           
            print("OOObject detection multi")
           
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            # print(names)
            
            
            keys =[]
            for v in cknames:
                value = [i for i in names if names[i]==v]
                keys.extend(value)
                
              
            # keys=keys.index(cknames)
           
            
            print(keys) 
            values = list( map(names.get, keys) )
            
            pred = self.model(frame, augment='store_true')
           
            
            # object.print()  # print results to screen
            objects = pred.xyxy[0]
            labels =  self.model.module.names if hasattr(self.model, 'module') else self.model.names
            for i in range(len(objects)):
                obj = objects[i]
                print(obj[5])
                if obj[5] in keys:
                    print("ooooo",obj)
                    label =labels[int(obj[5])]
                    print(label)
                    cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(obj[0]), int(obj[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
            # read image as BGR
            img_BGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #BGR
            frame = cv2.imencode('.jpg', img_BGR )[1].tobytes()
            return frame
           

def gen_det_obj_4multi(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_det_try_4multi')
def det_video_feed_try_4multi():
    url = request.args.get('url')
    return Response(gen_det_obj_4multi(Objdetection_4multi(url)), mimetype='multipart/x-mixed-replace; boundary=frame')
    







@app.route('/try_4multi', methods = ['POST', 'GET'])
def stream_view_4multi():
    global Row1,cknames
    rows = []
    cknames = request.form.getlist('skills')
    print(cknames)
 
    return render_template("home/try_multi.html",data=object_list,len = len(object_list),User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))





#####################################################################################################################################################
                        # Multi feed 
#####################################################################################################################################################

import cv2
import time
import torch
import subprocess as sp
import ffmpegfeed.feeds as fct 

@app.route('/multi_model_excution')
def multi_model_excution():
 
    
    return render_template('home/detection.html',User_Models_record=User_Models_record.query.filter_by(username=current_user.username))




class multifeed1():
    def __init__(self, url):
    
        # self.video = cv2.VideoCapture('rtmp://media1.ambicam.com:1938/dvr7/fd9b67cc-6c2e-46c6-99c4-0e13ac403e32')
        current_loggin_user=current_user.username
        fetch_url =  User_camera_sources.query.filter_by(username=current_loggin_user).first()
        print(fetch_url.link1)
        self.video = cv2.VideoCapture(fetch_url.link1)
        frame_width = int(self.video.get(3))
        frame_height = int(self.video.get(4))
        rtmp_url = "rtmp://media5.ambicam.com:1938/live/feed22222"
        ffmpeg = "ffmpeg -f rawvideo -pix_fmt bgr24 -s {}x{} -r 15 -i - -pix_fmt yuv420p -vcodec libx264 -preset ultrafast -tune zerolatency -http_persistent 0 -ar 8K -f flv {}".format(
            frame_width, frame_height, rtmp_url)
        self.url = url
        self.error_count = 0
        self.process = sp.Popen(ffmpeg.split(), stdin=sp.PIPE)
        self.model = torch.hub.load('yolov5', 'custom', path='/home/torquehqio/torquehq-io/main/Torque-AI/Users_slab/test/Models/coffeBottle.pt', source='local',_verbose=False, force_reload=True)

       

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
       

        
        # Set Model Settings Dynamic
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
        success, frame = self.video.read()
        if success == True:
            new_img = fct.draw_anchor_box(
                frame, fct.detection(frame, self.model))

            # ret,buffer=cv2.imencode('.jpg',frame)
            # frame=buffer.tobytes()
            
            # #print(type(frame))

            # img = Image.open(io.BytesIO(frame))
            # results = self.model(img, size=640)
        
            # results.print()  # print results to screen
            
            
            # #convert remove single-dimensional entries from the shape of an array
            # img = np.squeeze(results.render()) #RGB
            # # read image as BGR
            # img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            # frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            self.process.stdin.write(frame.tobytes())
            return frame
       

def gen_multifeed1(camera):
    while True:
        frame = camera.get_frame()
        # yield (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_multifeed1')
def video_feed_multifeed1():
    selectValue = request.form.get('model_name')
    print(selectValue)
    url = request.args.get('url')
    
    return Response(gen_multifeed1(multifeed1(url)), mimetype='multipart/x-mixed-replace; boundary=frame')

###############################################################################################################
class multifeed2():
    def __init__(self, url):
        # self.video = cv2.VideoCapture('rtmp://media1.ambicam.com:1938/dvr7/fd9b67cc-6c2e-46c6-99c4-0e13ac403e32')
        current_loggin_user=current_user.username
        fetch_url =  User_camera_sources.query.filter_by(username=current_loggin_user).first()
        print(fetch_url.link1)
        self.video = cv2.VideoCapture(fetch_url.link1)
        frame_width = int(self.video.get(3))
        frame_height = int(self.video.get(4))
        rtmp_url = "rtmp://media5.ambicam.com:1938/live/feed111111"
        ffmpeg = "ffmpeg -f rawvideo -pix_fmt bgr24 -s {}x{} -r 15 -i - -pix_fmt yuv420p -vcodec libx264 -preset ultrafast -tune zerolatency -http_persistent 0 -ar 8K -f flv {}".format(
            frame_width, frame_height, rtmp_url)
        self.url = url
        self.error_count = 0
        self.process = sp.Popen(ffmpeg.split(), stdin=sp.PIPE)
        self.model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local',_verbose=False, force_reload=True)

       

    def __del__(self):
        self.video.release()
    
    def get_frame(self):
       

        
        # Set Model Settings Dynamic
        self.model.eval()
        self.model.conf = 0.6  # confidence threshold (0-1)
        self.model.iou = 0.45  # NMS IoU threshold (0-1) 
      
        success, frame = self.video.read()
        if success == True:
            new_img = fct.draw_anchor_box(
                frame, fct.detection(frame, self.model))

            # ret,buffer=cv2.imencode('.jpg',frame)
            # frame=buffer.tobytes()
            
            # #print(type(frame))

            # img = Image.open(io.BytesIO(frame))
            # results = self.model(img, size=640)
        
            # results.print()  # print results to screen
            
            
            # #convert remove single-dimensional entries from the shape of an array
            # img = np.squeeze(results.render()) #RGB
            # # read image as BGR
            # img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
            # frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
            self.process.stdin.write(frame.tobytes())
            return frame
       

def gen_multifeed2(camera):
    while True:
        frame = camera.get_frame()
        # yield (b'--frame\r\n'
        #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_multifeed2')
def video_feed_multifeed2():
    url = request.args.get('url')
    
    return Response(gen_multifeed2(multifeed2(url)), mimetype='multipart/x-mixed-replace; boundary=frame')


#####################################################################################################################################################
###############    Segementation anything   ################################################################
import os
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
# from flask_bootstrap import Bootstrap


from segmentation.src.VideoStream import *




model_config = {
    "model_path": './segmentation/models/yolov8n-seg.onnx', # model path
    "classes_path" : 'segmentation/models/coco_label.txt', # classes path
    "box_score" : 0.4,
    "box_nms_iou" : 0.45,
    "box_aspect_ratio" : None,
    "box_stretch" : None,
}





@app.route('/segmentation',methods=["GET","POST"])
def seg():
    global VIDEO 
    TITLE = 'Object Segmentation App'
    current_loggin_user=current_user.username
    fetch_url =  User_camera_sources.query.filter_by(username=current_loggin_user).first()
    print(fetch_url.link1)
  
    cam_config = {
        "cam_id" :fetch_url.link1 ,
        'exposure': -2, # init cam exposure
        'contrast': 50 # init cam contrast
    }
    VIDEO = VideoStreaming(cam_config=cam_config, model_config=model_config)
    CAM_CONFIG = cam_config.copy()
    CAM_CONFIG["height"] = int(VIDEO.H)
    CAM_CONFIG["width"] = int(VIDEO.W)

    MODLE_CONFIG = model_config.copy()
    for key, value in model_config.items():
        if type(value) == str:
            MODLE_CONFIG[key] = os.path.basename(value)

    CLASSES_CONFIG = VIDEO.MODEL.colors_dict.copy()
    STYLE_CONFIG = VIDEO.style_dict.copy()
    return render_template('home/segmentation.html', TITLE=TITLE, 
                                        CAM_CONFIG = CAM_CONFIG, 
                                        MODEL_CONFIG = MODLE_CONFIG, 
                                        TARGETLIST =CLASSES_CONFIG  ,
                                        STYLELIST = STYLE_CONFIG)
 
@app.route('/video_feed')
def video_feed():
    '''
    Video streaming route.
    '''
    return Response(
        VIDEO.show(),
             mimetype='multipart/x-mixed-replace; boundary=frame'
    )



@app.route('/request_target_display')
def request_target_display():
    targets_list  = request.args.get('targetList')
    print('*'*10)
    print("display targets :", targets_list)
    print('*'*10)
    VIDEO.setViewTarget(targets_list) 

    return "nothing"   
# Button requests called from ajax
@app.route('/request_preview_switch')
def request_preview_switch():
    active  = request.args.get('active')
    VIDEO.preview = active
    print('*'*10)
    print("display preview :", VIDEO.preview)
    print('*'*10)
    return "nothing"

@app.route('/request_background_video')
def request_background_video():
    # url = "https://youtu.be/LtrtLL_8mLM" # testing url
    url  = request.args.get('url')
    print('*'*10)
    print("video url or path :", url)
    print('*'*10)
    VIDEO.setBackGround(url)
    return "nothing"

@app.route('/request_background_switch')
def request_background_switch():
    active  = request.args.get('active')
    VIDEO.background = active
    print('*'*10, VIDEO.background)
    return "nothing"

@app.route('/request_flipH_switch')
def request_flipH_switch():
    active  = request.args.get('active')
    VIDEO.flipH = active
    print('*'*10)
    print("display flip :",  VIDEO.flipH)
    print('*'*10)
    return "nothing"

@app.route('/request_model_switch')
def request_model_switch():
    type  = request.args.get('type')
    VIDEO.detect = type
    print('*'*10)
    print("display type :",  type)
    print('*'*10)
    return "nothing"

@app.route('/request_style_switch')
def request_style_switch():
    type  = request.args.get('type')
    VIDEO.setViewStyle(type)
    print('*'*10)
    print("display style :",  type)
    print('*'*10)
    return "nothing"

@app.route('/request_exposure')
def request_exposure():
    value  = request.args.get('value')
    VIDEO.exposure = int(value)
    print('*'*10)
    print("display exposure :", VIDEO.exposure)
    print('*'*10)
    return "nothing"


@app.route('/request_contrast')
def request_contrast():
    value  = request.args.get('value')
    VIDEO.contrast = int(value)
    print('*'*10)
    print("display contrast :",VIDEO.contrast)
    print('*'*10)
    return "nothing"

@app.route('/request_blur')
def request_blur():
    value  = request.args.get('value')
    VIDEO.blur = int(value)
    print('*'*10)
    print("display blur (kernel):",VIDEO.blur)
    print('*'*10)
    return "nothing"

@app.route('/reset_camera')
def reset_camera():
    STATUS =VIDEO.InitCamSettings()
    active  = request.args.get('active')
    VIDEO.flipH = active
    type  = request.args.get('type')
    VIDEO.detect = type
    print('*'*10)
    print("reset :",STATUS)
    print('*'*10)
    return "nothing"



if __name__ == "__main__":
   
    app.run(host="0.0.0.0")
    
    
