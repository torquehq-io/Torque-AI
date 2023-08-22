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

from flask import Flask, render_template, request, json, session, Response, url_for, send_file, redirect,stream_with_context,escape,flash
import os, base64, random
import  datetime
from os.path import join, dirname, realpath

import time
from datetime import date
import threading

from pathlib import Path


import os.path
import sys
import subprocess
import torch
import glob
import cv2
import shutil
import inspect
import mysql.connector
# subprocess.Popen(['gnome-terminal', '-e', 'python3 apitest.py'])
# subprocess.Popen(['gnome-terminal', '-e', 'python3 boothapi.py'])

# def ffmpegfeedall():
    # subprocess.Popen(['python3','feed1.py'])
    #subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed1.py'])
    # subprocess.Popen(['gnome-terminal', '-e', 'python3 ffmpegfeed/feed2.py'])
    # subprocess.Popen(['gnome-terminal', '-e', 'python3 apitest.py'])
    # subprocess.Popen(['gnome-terminal', '-e', 'python3 boothapi.py'])
    
# ffmpegfeedall()

#####################################################

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

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="raptorraptorraptorraptor0",
    database="flask_db",
    auth_plugin='mysql_native_password'
)
mycursor = mydb.cursor()


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








##############################################################################





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

########################################
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
import cvzone
class VideoPeopleDetection():
    time_reference = datetime.datetime.now()
    counter_frame = 0
    processed_fps = 0
   
    def __init__(self,url):

       
        self.model = YOLO('yolov8s.pt')
        self.classes = self.model.names
        self.url=url
        self.video_name = self.url
        self.people_count_history = []
        self.current_loggin_user=current_user.username

        self.last_capture_time = datetime.datetime.now()
        self.csv_file = os.getcwd()+"/Users_slab/"+self.current_loggin_user+"/crowd_counting_history/people_count_history.csv"  # CSV file to store the history
        self.initialize_csv_file()
        self.last_capture_time = datetime.datetime.now()  # Initialize the last capture time
        self.prev_people_count=-1
        self.cap = cv2.VideoCapture(self.video_name)

    def __del__(self):
        self.cap.release()

    def load_model(self, model_name):
        if model_name:
            self.model = torch.hub.load('yolov5', 'custom', path=self.modelName, source='local',_verbose=False, force_reload=True)
            return self.model
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def initialize_csv_file(self):
        # Create or overwrite the CSV file with header
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # writer.writerow(['Timestamp', 'Count', 'Image Path'])
            writer.writerow(['cameradid', 'sendtime', 'imgurl', 'an_id', 'ImgCount'])

    def save_data_to_csv(self, cameraid, timestamp, image_path, analyticsid, count ):
        
        if count >= 1:
            data = {
                'cameraid': cameraid,
                'timestamp': timestamp,
                'count': count,
                'image_path': image_path,
                'analyticsid': analyticsid
            }

            file_exists = os.path.isfile(self.csv_file)

            with open(self.csv_file, 'a', newline='') as csv_file:
               
                fieldnames = ['cameraid', 'timestamp', 'image_path', 'analyticsid', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(data)

    
    def get_frame(self):
        ret, frame = self.cap.read()
        my_file = open("crowd_counting/coco.txt", "r")
        data = my_file.read()
        class_list = data.split("\n")
        count = 0
        # Clear the CSV file if it exists
       
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()

        count += 1
        
        frame = cv2.resize(frame, (1020, 500))

        results = self.model.predict(frame)
        a = results[0].boxes.boxes
        a = a.cpu()  # Convert CUDA tensor to CPU tensor
        px = pd.DataFrame(a).astype("float")
        people_count = 0  # Reset people count for each frame
        for index, row in px.iterrows():
            x1 = int(a[index][0])
            y1 = int(a[index][1])
            x2 = int(a[index][2])
            y2 = int(a[index][3])
            d = int(a[index][5])
            c = class_list[d]
            if 'person' in c:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                people_count += 1  # Increment people count for each person detected

        cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        if people_count >= 1:
            if people_count != self.prev_people_count:
                # Capture and save the image
                camera_id = "PQRS-230651-ABCDE"
                image_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + camera_id +".jpg"
                image_path = "/home/torqueai/blobdrive/" + image_name 
                #image_name = "/home/torqueai/blobdrive" + image_name + ".jpg"
                cv2.imwrite(image_path, frame)
                print(f"Image captured: {image_name}")

                # Prepare data for CSV
                camera_id = "PQRS-230651-ABCDE"
                send_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                img_url = "https://inferenceimage.blob.core.windows.net/inferenceimages/" + image_name
                an_id = 1
                img_count = people_count

                # Save data to CSV
                data = {'cameradid': camera_id, 'sendtime': send_time, 'imgurl': img_url,
                        'an_id': an_id, 'ImgCount': img_count}
                self.save_data_to_csv(camera_id, send_time, img_url, an_id, img_count)
            self.prev_people_count = people_count
            # df = pd.DataFrame(data)
            # df.to_csv('people_count.csv', mode='a', header=not os.path.exists('people_count.csv'), index=False)

            # Send data to URL
          
        ret, jpeg = cv2.imencode(".jpg", frame)

        return jpeg.tobytes()
def gen_crowd_counting(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')



    
@app.route("/video_feed_for_crowd_counting")
def video_feed_for_crowd_counting():
    url = request.args.get('url')
    video_detector = VideoPeopleDetection(url)
    return Response(gen_crowd_counting(video_detector),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/crowd_counting.html' ,methods=["POST","GET"])
def crowd_counting():
  
    return render_template('home/crowd_counting.html', User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))
import pandas as pd




from flask import Response
import requests
import time
# API_ENDPOINT='https://analytics.vmukti.com/api/analytics'
@app.route('/people_count_data')
def get_data():
    current_loggin_user = current_user.username
    csv_file = os.path.join(os.getcwd(), 'Users_slab', current_loggin_user, 'crowd_counting_history', 'people_count_history.csv')
    
   
    # Prepare the data for plotting (using the same data structure as before)
    data = {
        'timestamp': [],
        'count': []
    }
    
    # Read data from the CSV file again to populate 'data' for plotting
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Append data from each row to the respective lists
            data['timestamp'].append(row.get('sendtime', ''))
            data['count'].append(int(row.get('ImgCount', 0)))

    # Convert the data to JSON
    json_data = json.dumps(data)

    # Set the cache-control header to disable caching
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Content-Type': 'application/json'
    }

    # Return the data as a JSON response with the cache-control headers
    return Response(json_data, headers=headers)




#################################### People counting booth demo ####################################################################
import pygame
class VideoPeopleDetection1():
    time_reference = datetime.datetime.now()
    counter_frame = 0
    processed_fps = 0
   
    def __init__(self,url):
        # Load YOLOv5 model
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.modelName = "crowdhuman_yolov5m.pt"
        self.model = self.load_model(self.modelName)
        self.classes = self.model.names
        self.url=url
        self.video_name = self.url
        self.people_count_history = []
        self.current_loggin_user=current_user.username
        self.roi_model=torch.hub.load('yolov5', 'custom', path="prebuilt_model/yolov5s.pt", source='local',_verbose=False, force_reload=True)
        self.last_capture_time = datetime.datetime.now()
        self.csv_file = os.getcwd()+"/Users_slab/"+self.current_loggin_user+"/crowd_counting_history/booth.csv"  # CSV file to store the history
        self.initialize_csv_file()
        self.last_capture_time = datetime.datetime.now()  # Initialize the last capture time
        # self.video_name = 'For_Validation6.mp4'

        # Read the video file
        self.cap = cv2.VideoCapture(self.video_name)

    def __del__(self):
        self.cap.release()

    def load_model(self, model_name):
        if model_name:
            self.model = torch.hub.load('yolov5', 'custom', path=self.modelName, source='local',_verbose=False, force_reload=True)
            return self.model
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def initialize_csv_file(self):
        # Create or overwrite the CSV file with header
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['cameradid', 'sendtime', 'imgurl', 'an_id', 'ImgCount'])

    def save_data_to_csv(self, cameraid, timestamp, image_path, analyticsid, count):
        if count > 1:
            data = {
                'cameraid': cameraid,
                'timestamp': timestamp,
                'count': count,
                'image_path': image_path,
                'analyticsid': analyticsid
            }

            file_exists = os.path.isfile(self.csv_file)

            with open(self.csv_file, 'a', newline='') as csv_file:
                # writer.writerow(['Timestamp', 'Count', 'Image Path'])
                # writer.writerow(['Camera ID', 'Time', 'URL', 'Analytics ID', 'Count'])
                # fieldnames = ['timestamp', 'count', 'image_path']
                fieldnames = ['cameraid', 'timestamp', 'image_path', 'analyticsid', 'count']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow(data)
    # def detect_chairs(self, frame):
    #     # Perform object detection with YOLOv5
    #     results = self.roi_model(frame, size=640)

    #     # Initialize an empty list to store chair coordinates
    #     chair_coordinates = []

    #     for obj in results.xyxy[0]:
    #         if obj[-1] == 56:  # 62 is the class ID for 'chair'
    #             # Extract chair coordinates
    #             xmin, ymin, xmax, ymax = map(int, obj[:4])
    #             chair_coordinates.append((xmin, ymin, xmax, ymax))

    #             # Draw bounding boxes around chairs
    #             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    #     return frame, chair_coordinates

    
    def get_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = self.cap.read()

        # Detect chairs and get chair coordinates
        # frame, chair_coordinates = self.detect_chairs(frame)

        # Generate ROI rectangle zone based on chair coordinates
        roi_x = 238
        roi_y = 148
        roi_width = 235
        roi_height = 159
############this for generating custome roi zone based on object position
        # if chair_coordinates:
        #     # Calculate the average coordinates of chairs
        #     avg_x = sum([x[0] for x in chair_coordinates]) / len(chair_coordinates)
        #     avg_y = sum([x[1] for x in chair_coordinates]) / len(chair_coordinates)
        #     avg_width = sum([x[2] - x[0] for x in chair_coordinates]) / len(chair_coordinates)
        #     avg_height = sum([x[3] - x[1] for x in chair_coordinates]) / len(chair_coordinates)

        #     # Add a certain distance (1 meter) to the coordinates
        #     roi_x = int(avg_x - avg_width / 2 - 100)
        #     roi_y = int(avg_y - avg_height / 2 - 100)
        #     roi_width = int(avg_width + 200)
        #     roi_height = int(avg_height + 200)

            # Draw the ROI rectangle
        

        num_people, alert_message = self.count_people_in_roi(frame, roi_x, roi_y, roi_width, roi_height)

        # Draw the ROI rectangle
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

        # Draw the number of people and alert message on the frame
        cv2.putText(frame, f'FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.putText(frame, f'People: {num_people}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Processed FPS: {VideoPeopleDetection1.processed_fps}', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, alert_message, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        current_time = datetime.datetime.now()
        time_diff = (current_time - self.last_capture_time).total_seconds()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Capture an image every 10 seconds
        if num_people > 1:
            # fieldnames = ['cameraid', 'timestamp', 'image_path', 'analyticsid', 'count']
            current_time = datetime.datetime.now()
            image_name = current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            # image_path = os.path.join(os.getcwd() + "/Users_slab/" + self.current_loggin_user + "/crowd_counting_history/images/", image_name)
            # writer.writerow(['Camera ID', 'Time', 'URL', 'Analytics ID', 'Count'])
            blobdrive_image_path = "/home/torqueai/blobdrive"
            image_path = os.path.join(blobdrive_image_path, image_name)
            cv2.imwrite(image_path, frame)
            imgurl = "https://inferenceimage.blob.core.windows.net/inferenceimages/" + image_name
            sendtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cameraid = self.video_name
            split_string = cameraid.split('/')
            cameradid = split_string[-1]
            an_id = 2
            ImgCount = num_people
            self.save_data_to_csv(cameradid, sendtime, imgurl, an_id, ImgCount)

            self.last_capture_time = current_time

            

        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()
    def count_people_in_roi(self, frame, roi_x, roi_y, roi_width, roi_height):
        # Crop the frame to the ROI
        roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Apply a person detection algorithm to detect people in the ROI (e.g., OpenCV's HOG-based person detector)
        results = self.model(roi_frame, size=640)
        
        # Initialize variables
        num_people = 0
        alert_message = ""

        # Loop through each detected object and count the people within the ROI
        for obj in results.xyxy[0]:
            if obj[-1] == 0:  # 0 is the class ID for 'person'
                # Extract person coordinates within the ROI
                xmin, ymin, xmax, ymax = map(int, obj[:4])

                num_people += 1

                # Draw a bounding box around the person within the ROI
                cv2.rectangle(frame, (roi_x + xmin, roi_y + ymin), (roi_x + xmax, roi_y + ymax), (0, 0, 255), 2)
                cv2.putText(frame, f"Person", (roi_x + xmin, roi_y + ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Check if the number of people exceeds the limit
        if num_people > 1:  # Adjust the limit as per your requirement
            alert_message = "Alert: Exceeded maximum limit of people!"

        # Return the number of people and the alert message
        return num_people, alert_message

    # def count_people_in_roi(self, frame, roi_x, roi_y, roi_width, roi_height):
    # # Crop the frame to the ROI
    #     roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    #     # Convert the ROI frame to grayscale

    #     # Apply a person detection algorithm to count the people in the ROI (e.g., OpenCV's HOG-based person detector)
    #     results = self.model(roi_frame, size=640)

    #     # Loop through each detected object and count the people
    #     num_people = 0
    #     bgr = (0, 255, 0)

    #     # To get the processed FPS
    #     # VideoPeopleDetection.time_reference = datetime.datetime.now()

    #     time_now = datetime.datetime.now()
    #     time_diff = (time_now - VideoPeopleDetection.time_reference).seconds

    #     if time_diff >= 1:
    #         VideoPeopleDetection.time_reference = datetime.datetime.now()
    #         VideoPeopleDetection.processed_fps = VideoPeopleDetection.counter_frame
    #         VideoPeopleDetection.counter_frame = 0
    #     else:
    #         VideoPeopleDetection.counter_frame += 1

    #     for obj in results.xyxy[0]:
    #         if obj[-1] == 0:  # 0 is the class ID for 'person'
    #             # Extract person coordinates
    #             xmin, ymin, xmax, ymax = map(int, obj[:4])
    #             accuracy = obj[4]

    #             # Check if the person coordinates intersect with the ROI rectangle
    #             if roi_x <= xmin <= roi_x + roi_width and roi_y <= ymin <= roi_y + roi_height and \
    #             roi_x <= xmax <= roi_x + roi_width and roi_y <= ymax <= roi_y + roi_height:
    #                 if accuracy > 0.5:
    #                     num_people += 1
    #                     # Append people count and timestamp to history
    #                     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                     self.people_count_history.append({
    #                         'timestamp': timestamp,
    #                         'count': num_people
    #                     })

    #                     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #                     cv2.putText(frame, f" {round(float(accuracy), 2)}", (xmin, ymin),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #     # Draw the number of people on the frame and display it

    #     # Return the number of people detected in the ROI
    #     return num_people

    # def count_people_in_roi(self, frame, roi_x, roi_y, roi_width, roi_height):
    #     # Crop the frame to the ROI
    #     roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]

    #     # Convert the ROI frame to grayscale

    #     # Apply a person detection algorithm to count the people in the ROI (e.g., OpenCV's HOG-based person detector)
    #     results = self.model(frame, size=640)

    #     # Loop through each detected object and count the people
    #     num_people = 0
    #     bgr = (0, 255, 0)

    #     # To get the processed FPS
    #     # VideoPeopleDetection.time_reference = datetime.datetime.now()

    #     time_now = datetime.datetime.now()
    #     time_diff = (time_now - VideoPeopleDetection.time_reference).seconds

    #     if time_diff >= 1:
    #         VideoPeopleDetection.time_reference = datetime.datetime.now()
    #         VideoPeopleDetection.processed_fps = VideoPeopleDetection.counter_frame
    #         VideoPeopleDetection.counter_frame = 0
    #     else:
    #         VideoPeopleDetection.counter_frame += 1

    #     for obj in results.xyxy[0]:
    #         if obj[-1] == 0:  # 0 is the class ID for 'person'
    #             # Extract person coordinates
    #             xmin, ymin, xmax, ymax = map(int, obj[:4])
    #             accuracy = obj[4]

    #             # Check if the person coordinates intersect with the ROI rectangle
    #             if roi_x <= xmin <= roi_x + roi_width and roi_y <= ymin <= roi_y + roi_height and \
    #             roi_x <= xmax <= roi_x + roi_width and roi_y <= ymax <= roi_y + roi_height:
    #                 if accuracy > 0.5:
    #                     num_people += 1
    #                     # Append people count and timestamp to history
    #                     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #                     self.people_count_history.append({
    #                         'timestamp': timestamp,
    #                         'count': num_people
    #                     })

    #                     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #                     cv2.putText(frame, f" {round(float(accuracy), 2)}", (xmin, ymin),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    #     # Draw the number of people on the frame and display it
    #     cv2.putText(frame, f'People: {num_people}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #     # Check if the number of people exceeds the limit
    #     if num_people > 1:
    #         flash("Alert: Too many people detected!",'Alert')
    #         # Generate an alert by playing a sound
    #         # pygame.mixer.init()
    #         # alert_sound = pygame.mixer.Sound('alert_sound.wav')
    #         # alert_sound.play()

    #     # Return the number of people detected in the ROI
    #     return num_people
def gen_crowd_counting1(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame
               + b'\r\n\r\n')



    
@app.route("/video_feed_for_crowd_counting1")
def video_feed_for_crowd_counting1():
    url = request.args.get('url')
    video_detector = VideoPeopleDetection1(url)

   
    return Response(gen_crowd_counting1(video_detector),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/crowd_counting1.html' ,methods=["POST","GET"])
def crowd_counting1():
  
    return render_template('home/crowd_counting1.html', User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))
import pandas as pd




from flask import Response
import time
# API_ENDPOINT='https://analyticsapi.vmukti.com/api/Analytics'
@app.route('/people_count_data1')
def get_data1():
    current_loggin_user=current_user.username
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(os.getcwd()+"/Users_slab/"+current_loggin_user+"/crowd_counting_history/booth.csv")
    csv_file = os.path.join(os.getcwd(), 'Users_slab', current_loggin_user, 'crowd_counting_history', 'booth.csv')
    # Update the following lines with the correct column names
    # timestamp_column = 'sendtime'
    # count_column = 'ImgCount'

    # # Prepare the data for plotting
    # data = {
    #     'timestamp': df[timestamp_column].tolist(),
    #     'count': df[count_column].tolist()
    # }

    # # Convert the data to JSON
    # json_data = json.dumps(data)

    

    # Set the cache-control header to disable caching
    data = {
        'timestamp': [],
        'count': []
    }
    
    # Read data from the CSV file again to populate 'data' for plotting
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Append data from each row to the respective lists
            data['timestamp'].append(row.get('sendtime', ''))
            data['count'].append(int(row.get('ImgCount', 0)))

    # Convert the data to JSON
    json_data = json.dumps(data)

    # Set the cache-control header to disable caching
    headers = {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
        'Content-Type': 'application/json'
    }

    # Return the data as a JSON response with the cache-control headers
    return Response(json_data, headers=headers)
##########################################################################################

##############beep counting####################
# import ffmpeg
# import librosa
# import numpy as np
# from scipy.io.wavfile import write
# import os

# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from skimage.transform import resize
# class Hls_Link_extractor():
#     def __init__(self):
#         self.input_url="https://vm4inj5.vmukti.com:443/live-record/SSAE-425389-AFFDA.m3u8"
#         self.out_audio = "/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac"
#     def audio_stream_to_new_file(self):
#         try:
#             # Copy audio stream to a new file
#             (
#                 ffmpeg
#                 .input(self.input_url)
#                 .output(self.out_audio , acodec='aac', ab='16k')  # convert audio stream to AAC with a lower bit rate
#                 .run(capture_stdout=True, capture_stderr=True)
#             )

#         except ffmpeg.Error as e:
#             print('stdout:', e.stdout.decode('utf8'))
#             print('stderr:', e.stderr.decode('utf8'))
#             raise e

# class Segmentor():
#     def segment_audio(audio_path, timestamps, output_dir):
#         y, sr = librosa.load(audio_path, sr=22050)

#         for i, (start, end, label) in enumerate(timestamps):
#             start_sample = int(start * sr)
#             end_sample = int(end * sr)

#             segment = y[start_sample:end_sample]
#             output_path = f"{output_dir}/segment_{i}_{label}.wav"
#             write(output_path, sr, segment.astype(np.float32))

#     def convert_to_spectrogram(audio_path, n_fft=2048):
#         y, sr = librosa.load(audio_path, sr=22050)
        
#         # If the audio is shorter than n_fft, pad it with zeros
#         if len(y) < n_fft:
#             y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
        
#         spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
#         spectrogram = librosa.power_to_db(spectrogram)
#         return spectrogram


#     def segmen():
#         audio_path = '/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac'
#         output_dir = '/home/torque/github/main/Torque-AI/beep_counter/data/segments'
#         timestamps = [
#             (0, 3, 'beep'),
#             (5, 39, 'non-beep'),
#             (40, 43, 'beep'),
#             (48, 102, 'non-beep'),
#             (103, 106, 'beep'),
#             (110, 125, 'non-beep'),
#             (128, 134, 'beep'),
#             (138, 158, 'non-beep')
#         ]



#         Segmentor.segment_audio(audio_path, timestamps, output_dir)

#         for i, (start, end, label) in enumerate(timestamps):
#             segment_path = f"{output_dir}/segment_{i}_{label}.wav"
#             spectrogram = Segmentor.convert_to_spectrogram(segment_path)
#             # Now `spectrogram` is a 2D array representing the spectrogram of the audio segment.

# class Label():
#     def convert_to_spectrogram(audio_path, n_fft=2048):
#         y, sr = librosa.load(audio_path, sr=22050)
        
#         # If the audio is shorter than n_fft, pad it with zeros
#         if len(y) < n_fft:
#             y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
        
#         spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
#         spectrogram = librosa.power_to_db(spectrogram)
#         return spectrogram
#     def model_prep():
#         # Prepare data
#         spectrograms = [Label.convert_to_spectrogram(os.path.join('/home/torque/github/main/Torque-AI/beep_counter/data/segments', f)) for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')]
#         spectrograms_resized = [resize(s, (128, 128, 1)) for s in spectrograms]
#         X = np.array(spectrograms_resized)
#         y = np.array([0 if 'beep' in f else 1 for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')])  # Encode labels (0: beep, 1: non-beep)

#         # Split data into training set and test set
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Convert labels to one-hot encoding
#         y_train = to_categorical(y_train, num_classes=2)
#         y_test = to_categorical(y_test, num_classes=2)

#         # Define model
#         model = Sequential([
#             Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
#             MaxPooling2D((2, 2)),
#             Conv2D(64, (3, 3), activation='relu'),
#             MaxPooling2D((2, 2)),
#             Flatten(),
#             Dense(64, activation='relu'),
#             Dropout(0.5),
#             Dense(2, activation='softmax')
#         ])

#         # Compile model
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#         # Train model
#         model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

#         # Evaluate model
#         test_loss, test_acc = model.evaluate(X_test, y_test)
#         print('Test accuracy:', test_acc)

#         # Save the model
#         # Save the model in SavedModel format
#         model.save('/home/torque/github/main/Torque-AI/beep_counter/trained_model')
# ############
# @app.route('/beep_counter',methods=['POST','GET'])
# def beep_counter():
#     # Create instances of your classes and execute their methods here
#     hls_extractor = Hls_Link_extractor()
#     hls_extractor.audio_stream_to_new_file()
    
#     segmentor = Segmentor()
#     segmentor.segment_audio(segmentor.audio_path, segmentor.timestamps, segmentor.output_dir)
    
#     label = Label()
#     label.convert_to_spectrogram(label.audio_path)
#     return "model generated"
# @app.route('/beep_count')
# def beepcnt():
    
   
#     csv_file = os.path.join(os.getcwd(), 'beep_count.csv')
   
#     data = {
#         'timestamp': [],
#         'count': [],
#         'total_count':[]
#     }
    
#     # Read data from the CSV file again to populate 'data' for plotting
#     with open(csv_file, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             # Append data from each row to the respective lists
#             data['timestamp'].append(row.get('Timestamp', ''))
#             data['count'].append(int(row.get('BeepCount', 0)))
#             data['total_count'].append(int(row.get('TotalBeepCount', 0)))

#     # Convert the data to JSON
#     json_data = json.dumps(data)

#     # Set the cache-control header to disable caching
#     headers = {
#         'Cache-Control': 'no-cache, no-store, must-revalidate',
#         'Pragma': 'no-cache',
#         'Expires': '0',
#         'Content-Type': 'application/json'
#     }

#     # Return the data as a JSON response with the cache-control headers
#     return Response(json_data, headers=headers)
# # import m3u8
# # @app.route('/hls_stream')
# # def hls_stream():
# #     # HLS URL
# #     hls_url = 'https://t1.arcischain.io:8443/live/5/index.m3u8'
    
# #     # Parse the HLS manifest
# #     m3u8_obj = m3u8.load(hls_url)
    
# #     # Try to find the playlist URL (modify this based on your HLS structure)
# #     playlist_url = None
    
# #     for playlist in m3u8_obj.playlists:
# #         if playlist.stream_info and 'BANDWIDTH' in playlist.stream_info:
# #             playlist_url = playlist.absolute_uri
# #             break

# #     if playlist_url is None:
# #         return "No valid playlist found in the HLS manifest."

# #     return render_template('hls_stream.html', playlist_url=playlist_url)

    
# subprocess.Popen(['gnome-terminal', '-e', 'python3  /home/torque/github/main/Torque-AI/beep_counter/audio_predictor_extractor_live.py'])
# subprocess.Popen(['gnome-terminal', '-e', 'python3  /home/torque/github/main/Torque-AI/beep_counter/audio_predictor_apel_opt.py'])  
# ###################################################
############face recognition######################

from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from skimage.transform import resize
class Hls_Link_extractor():
    def __init__(self):
        self.input_url="https://vm4inj5.vmukti.com:443/live-record/SSAE-425389-AFFDA.m3u8"
        self.out_audio = "/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac"
    def audio_stream_to_new_file(self):
        try:
            # Copy audio stream to a new file
            (
                ffmpeg
                .input(self.input_url)
                .output(self.out_audio , acodec='aac', ab='16k')  # convert audio stream to AAC with a lower bit rate
                .run(capture_stdout=True, capture_stderr=True)
            )

        except ffmpeg.Error as e:
            print('stdout:', e.stdout.decode('utf8'))
            print('stderr:', e.stderr.decode('utf8'))
            raise e

class Segmentor():
    def segment_audio(audio_path, timestamps, output_dir):
        y, sr = librosa.load(audio_path, sr=22050)

        for i, (start, end, label) in enumerate(timestamps):
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            segment = y[start_sample:end_sample]
            output_path = f"{output_dir}/segment_{i}_{label}.wav"
            write(output_path, sr, segment.astype(np.float32))

    def convert_to_spectrogram(audio_path, n_fft=2048):
        y, sr = librosa.load(audio_path, sr=22050)
        
        # If the audio is shorter than n_fft, pad it with zeros
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
        
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram)
        return spectrogram


    def segmen():
        audio_path = '/home/torque/github/main/Torque-AI/beep_counter/out_audio.aac'
        output_dir = '/home/torque/github/main/Torque-AI/beep_counter/data/segments'
        timestamps = [
            (0, 3, 'beep'),
            (5, 39, 'non-beep'),
            (40, 43, 'beep'),
            (48, 102, 'non-beep'),
            (103, 106, 'beep'),
            (110, 125, 'non-beep'),
            (128, 134, 'beep'),
            (138, 158, 'non-beep')
        ]



        Segmentor.segment_audio(audio_path, timestamps, output_dir)

        for i, (start, end, label) in enumerate(timestamps):
            segment_path = f"{output_dir}/segment_{i}_{label}.wav"
            spectrogram = Segmentor.convert_to_spectrogram(segment_path)
            # Now `spectrogram` is a 2D array representing the spectrogram of the audio segment.

class Label():
    def convert_to_spectrogram(audio_path, n_fft=2048):
        y, sr = librosa.load(audio_path, sr=22050)
        
        # If the audio is shorter than n_fft, pad it with zeros
        if len(y) < n_fft:
            y = np.pad(y, (0, n_fft - len(y)), constant_values=(0, 0))
        
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        spectrogram = librosa.power_to_db(spectrogram)
        return spectrogram
    def model_prep():
        # Prepare data
        spectrograms = [Label.convert_to_spectrogram(os.path.join('/home/torque/github/main/Torque-AI/beep_counter/data/segments', f)) for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')]
        spectrograms_resized = [resize(s, (128, 128, 1)) for s in spectrograms]
        X = np.array(spectrograms_resized)
        y = np.array([0 if 'beep' in f else 1 for f in os.listdir('/home/torque/github/main/Torque-AI/beep_counter/data/segments')])  # Encode labels (0: beep, 1: non-beep)

        # Split data into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        # Define model
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train model
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

        # Evaluate model
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('Test accuracy:', test_acc)

        # Save the model
        # Save the model in SavedModel format
        model.save('/home/torque/github/main/Torque-AI/beep_counter/trained_model')
############
@app.route('/beep_counter',methods=['POST','GET'])
def beep_counter():
    # Create instances of your classes and execute their methods here
    hls_extractor = Hls_Link_extractor()
    hls_extractor.audio_stream_to_new_file()
    
    segmentor = Segmentor()
    segmentor.segment_audio(segmentor.audio_path, segmentor.timestamps, segmentor.output_dir)
    
    label = Label()
    label.convert_to_spectrogram(label.audio_path)
    return "model generated"



####################################################################################################################################
############face recognition######################

from flask import Flask, render_template, request, session, redirect, url_for, Response, jsonify
import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
 

 
cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="raptorraptorraptorraptor0",
    database="flask_db",
    auth_plugin='mysql_native_password'
)
mycursor = mydb.cursor()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Generate dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def generate_dataset(nbr):
    face_classifier = cv2.CascadeClassifier("face_recogniser/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    mycursor.execute("select ifnull(max(img_id), 0) from img_dataset")
    row = mycursor.fetchone()
    lastid = row[0]
 
    img_id = lastid
    max_imgid = img_id + 100
    count_img = 0
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
            img_id += 1
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "face_recogniser/dataset/"+nbr+"."+ str(img_id) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            mycursor.execute("""INSERT INTO `img_dataset` (`img_id`, `img_person`) VALUES
                                ('{}', '{}')""".format(img_id, nbr))
            mydb.commit()
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13 or int(img_id) == int(max_imgid):
                break
                cap.release()
                cv2.destroyAllWindows()
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Train Classifier >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
@app.route('/train_classifier/<nbr>')
def train_classifier(nbr):
    dataset_dir = "face_recogniser/dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
 
    return redirect('/facial_recognition')
 
 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Face Recognition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def face_recognition():  # generate frame by frame from camera
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        global justscanned
        global pause_cnt
 
        pause_cnt += 1
 
        coords = []
 
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
 
            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1
 
                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w
 
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
 
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
 
                mycursor.execute("select a.img_person, b.prs_name, b.prs_skill "
                                 "  from img_dataset a "
                                 "  left join prs_mstr b on a.img_person = b.prs_nbr "
                                 " where img_id = " + str(id))
                row = mycursor.fetchone()
                pnbr = row[0]
                pname = row[1]
                pskill = row[2]
 
                if int(cnt) == 30:
                    cnt = 0
 
                    mycursor.execute("insert into accs_hist (accs_date, accs_prsn) values('"+str(date.today())+"', '" + pnbr + "')")
                    mydb.commit()
 
                    cv2.putText(img, pname + ' | ' + pskill, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    time.sleep(1)
 
                    justscanned = True
                    pause_cnt = 0
 
            else:
                if not justscanned:
                    cv2.putText(img, 'UNKNOWN', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
 
                if pause_cnt > 80:
                    justscanned = False
 
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("face_recogniser/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("face_recogniser/classifier.xml")
 
    wCam, hCam = 400, 400
 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)
 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
        key = cv2.waitKey(1)
        if key == 27:
            break
 
 
 
@app.route('/facial_recognition')
def facial_recognition():
    mycursor.execute("select prs_nbr, prs_name, prs_skill, prs_active, prs_added from prs_mstr")
    data = mycursor.fetchall()
 
    return render_template('home/face_recogniser.html', data=data)
 
@app.route('/addprsn')
def addprsn():
    mycursor.execute("select ifnull(max(prs_nbr) + 1, 101) from prs_mstr")
    row = mycursor.fetchone()
    nbr = row[0]
    # print(int(nbr))
 
    return render_template('home/addprsn.html', newnbr=int(nbr))
 
@app.route('/addprsn_submit', methods=['POST'])
def addprsn_submit():
    prsnbr = request.form.get('txtnbr')
    prsname = request.form.get('txtname')
    prsskill = request.form.get('optskill')
 
    mycursor.execute("""INSERT INTO `prs_mstr` (`prs_nbr`, `prs_name`, `prs_skill`) VALUES
                    ('{}', '{}', '{}')""".format(prsnbr, prsname, prsskill))
    mydb.commit()
 
    # return redirect(url_for('home'))
    return redirect(url_for('vfdataset_page', prs=prsnbr))
 
@app.route('/vfdataset_page/<prs>')
def vfdataset_page(prs):
    return render_template('home/gendataset.html', prs=prs)
 
@app.route('/vidfeed_dataset/<nbr>')
def vidfeed_dataset(nbr):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(nbr), mimetype='multipart/x-mixed-replace; boundary=frame')
 
 
@app.route('/video_feed_fr')
def video_feed_fr():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')
 
@app.route('/fr_page')
def fr_page():
    """Video streaming home page."""
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, a.accs_added "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return render_template('home/fr_page.html', data=data)
 
 
@app.route('/countTodayScan')
def countTodayScan():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="raptorraptorraptorraptor0",
        database="flask_db",
        
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select count(*) "
                     "  from accs_hist "
                     " where accs_date = curdate() ")
    row = mycursor.fetchone()
    rowcount = row[0]
 
    return jsonify({'rowcount': rowcount})
 
 
@app.route('/loadData', methods = ['GET', 'POST'])
def loadData():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="raptorraptorraptorraptor0",
        database="flask_db",
        auth_plugin='mysql_native_password'
    )
    mycursor = mydb.cursor()
 
    mycursor.execute("select a.accs_id, a.accs_prsn, b.prs_name, b.prs_skill, date_format(a.accs_added, '%H:%i:%s') "
                     "  from accs_hist a "
                     "  left join prs_mstr b on a.accs_prsn = b.prs_nbr "
                     " where a.accs_date = curdate() "
                     " order by 1 desc")
    data = mycursor.fetchall()
 
    return jsonify(response = data)

############################################################################################

import cv2
import numpy as np
import time
import torch
from paddleocr import PaddleOCR
import csv
import os

csv_filename = 'captured_data.csv'
image_save_path = 'apps/static/detected_images'

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Plate Number', 'OCR Text', 'Image Filename'])

def generate_frames():
    # ... (previous code)
    model_path = r"ANPR.pt"
    cpu_or_cuda = "cpu"
    device = torch.device(cpu_or_cuda)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model = model.to(device)
    capture = cv2.VideoCapture('1car.flv')

    text_font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 0, 255)
    text_font_scale = 1.25
    prev_frame_time = 0
    new_frame_time = 0

    ocr = PaddleOCR()
    while True:
        
        ret, image = capture.read()
        if ret:
            output = model(image)
            result = np.array(output.pandas().xyxy[0])
            for i in result:
                p1 = (int(i[0]), int(i[1]))
                p2 = (int(i[2]), int(i[3]))
                text_origin = (int(i[0]), int(i[1]) - 5)
                cv2.rectangle(image, p1, p2, color=color, thickness=2)
                cv2.putText(image, f"{i[-1]} {i[-3]:.2f}", org=text_origin, fontFace=text_font,
                            fontScale=text_font_scale, color=color, thickness=2)

                plate_image = image[p1[1]:p2[1], p1[0]:p2[0]]
                ocr_result = ocr.ocr(plate_image)
                ocr_text = ' '.join([word_info[1][0] for word_info in ocr_result[0]])
                cv2.putText(image, ocr_text, (p1[0], p2[1] + 20), text_font, text_font_scale,
                            (255, 0, 0), thickness=2)

                # Generate unique filename for the image
                image_filename = f"{time.time():.0f}.jpg"
                image_path = os.path.join(image_save_path, image_filename)

                # Save the detected license plate image
                cv2.imwrite(image_path, plate_image)

                # Store captured data in the CSV file
                with open(csv_filename, mode='a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow([i[-1], ocr_text, image_filename])

                # ... (rest of the code)
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            cv2.putText(image, fps, (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        else:
            break

    capture.release()



@app.route('/video_feed_ANPR')
def video_feed_ANPR():


    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# ... (remaining code)

@app.route('/get_data_ANPR')
def get_data_ANPR():
    generate_data = []

    for i, row in enumerate(csv.reader(open(csv_filename))):
        if i == 0:
            continue
        generate_data.append(row)

    table_html = ''
    for row in generate_data:
        table_html += f'<tr><td>{row[0]}</td><td>{row[1]}</td><td><img src="{url_for("static", filename="detected_images/" + row[2])}" width="100"></td></tr>'

    return table_html


############################################################################################
import cv2
import numpy as np
import time
import torch
from paddleocr import PaddleOCR
import csv
import os
csv_filename = 'captured_data.csv'
image_save_path = 'apps/static/detected_images'

if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Plate Number', 'OCR Text', 'Image Filename'])

class AnprDetection():
    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)
        self.url = url
        self.error_count = 0
        self.cpu_or_cuda = "cuda"
        self.device = torch.device(self.cpu_or_cuda)
        self.ocr = PaddleOCR()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='ANPR.pt', force_reload=True)
        
        # self.model = torch.hub.load('yolov5', 'custom', path=loadModel + '/yolov5/runs/train/'+gLabel+'/weights/'+gLabel+'.pt', source='local', force_reload=True)
    def __del__(self):
        self.capture.release()
    
    def get_frame_anpr(self):
        self.text_font = cv2.FONT_HERSHEY_PLAIN
        self.color = (0, 0, 255)
        self.text_font_scale = 1.25
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        while True:
        
            ret, image = self.capture.read()
            if ret:
                output = self.model(image)
                result = np.array(output.pandas().xyxy[0])
                for i in result:
                    p1 = (int(i[0]), int(i[1]))
                    p2 = (int(i[2]), int(i[3]))
                    text_origin = (int(i[0]), int(i[1]) - 5)
                    cv2.rectangle(image, p1, p2, color=self.color, thickness=2)
                    cv2.putText(image, f"{i[-1]} {i[-3]:.2f}", org=text_origin, fontFace=self.text_font,
                                fontScale=self.text_font_scale, color=self.color, thickness=2)

                    plate_image = image[p1[1]:p2[1], p1[0]:p2[0]]
                    ocr_result = self.ocr.ocr(plate_image)
                    ocr_text = ' '.join([word_info[1][0] for word_info in ocr_result[0]])
                    cv2.putText(image, ocr_text, (p1[0], p2[1] + 20), self.text_font, self.text_font_scale,
                                (255, 0, 0), thickness=2)

                    # Generate unique filename for the image
                    image_filename = f"{time.time():.0f}.jpg"
                    image_path = os.path.join(image_save_path, image_filename)

                    # Save the detected license plate image
                    cv2.imwrite(image_path, plate_image)

                    # Store captured data in the CSV file
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow([i[-1], ocr_text, image_filename])

                    # ... (rest of the code)
                self.new_frame_time = time.time()
                fps = 1 / (self.new_frame_time - self.prev_frame_time)
                self.prev_frame_time = self.new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(image, fps, (7, 70), self.text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', image)
                frame =  buffer.tobytes()
                return frame                


 


        
       

def gen_det_anpr(camera):
    while True:
        frame = camera.get_frame_anpr()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed_det')
def det_video_feed():
    url = request.args.get('url')
    return Response(gen_det_anpr(AnprDetection(url)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anprtest.html' ,methods=["POST","GET"])
def anprtest():
  
    return render_template('home/anprtest.html', User_camera_sources=User_camera_sources_record.query.filter_by(username=current_user.username))


############################################################################################
if __name__ == "__main__":
   
    app.run(host="0.0.0.0")
    
    
