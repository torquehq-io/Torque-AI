from mtcnn import MTCNN

from FR import face_model
import warnings
import sys
import dlib
# from src.insightface.deploy import face_model

warnings.filterwarnings('ignore')

from keras.models import load_model
import face_preprocess
import numpy as np
import pickle
import cv2

import os

class FacePredictor():
    def __init__(self):
      

        self.image_size = '112,112'
        self.model = os.path.sep.join(
            [str(os.getcwd()), "FR/mtcnn-model/model,0"])
        self.threshold = 1.24
        self.det = 0
        self.model_filename = os.path.sep.join(
            [str(os.getcwd()) ,"FR/com_in_ineuron_ai_sorting/model_data/mars-small128.pb"])

        # # Initialize detector
        self.detector = MTCNN()

        # Initialize faces embedding model
        self.embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        self.embeddings = os.path.sep.join(
            [str(os.getcwd()) ,"FR/faceEmbeddingModels/embeddings.pickle"])
        self.le = os.path.sep.join(
            [str(os.getcwd()) ,"FR/faceEmbeddingModels/le.pickle"])

        # Load embeddings and labels
        self.data = pickle.loads(open(self.embeddings, "rb").read())
        self.le = pickle.loads(open(self.le, "rb").read())

        self.embeddings = np.array(self.data['embeddings'])
        self.labels = self.le.fit_transform(self.data['names'])

        # Load the classifier model
        self.model = load_model(os.path.sep.join(
            [str(os.getcwd()) ,"FR/faceEmbeddingModels/my_model.h5"]))

        self.cosine_threshold = 0.8
        self.proba_threshold = 0.85
        self.comparing_num = 5

        # # Tracker params
        self.trackers = []
        self.texts = []

        self.cap = cv2.VideoCapture("rtmp://media5.ambicam.com:1938/live/81f911d4-84b2-43a9-98bd-bdba4dc4538f")
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        print(str(self.frame_width) + " : " + str(self.frame_height))
        self.save_width = 800
        self.save_height = int(800 / self.frame_width * self.frame_height)

        

    # Define distance function
    @staticmethod
    def findCosineDistance(vector1, vector2):
        """
        Calculate cosine distance between two vector
        """
        vec1 = vector1.flatten()
        vec2 = vector2.flatten()

        a = np.dot(vec1.T, vec2)
        b = np.dot(vec1.T, vec1)
        c = np.dot(vec2.T, vec2)
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    @staticmethod
    def CosineSimilarity(test_vec, source_vecs):
        """
        Verify the similarity of one vector to group vectors of one class
        """
        cos_dist = 0
        for source_vec in source_vecs:
            cos_dist += FacePredictor.findCosineDistance(test_vec, source_vec)
        return cos_dist / len(source_vecs)

    def detectFace(self):
        # Initialize some useful arguments
        cosine_threshold = 0.8
        proba_threshold = 0.85
        comparing_num = 5
        trackers = []
        texts = []
        frames = 0

        # Start streaming and recording
        # cap = cv2.VideoCapture("rtmp://media5.ambicam.com:1938/live/81f911d4-84b2-43a9-98bd-bdba4dc4538f")
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        # print(str(frame_width) + " : " + str(frame_height))
        # save_width = 800
        # save_height = int(800 / frame_width * frame_height)

        while True:
            ret, frame = self.cap.read()
            frames += 1
            frame = cv2.resize(frame, (self.save_width, self.save_height))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = "Running the prediction"
            cv2.putText(frame, text,
                        (int(0.3 * frame.shape[1]), int(0.05 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
            # if frames % 3 == 0:
            trackers = []
            texts = []

            bboxes =  self.detector.detect_faces(frame)

            if len(bboxes) != 0:

                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                            landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                            landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                            landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2, 0, 1))
                    embedding = self.embedding_model.get_feature(nimg).reshape(1, -1)

                    text = "Unknown"

                    # Predict class
                    preds =  self.model.predict(embedding)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = ( self.labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = self.embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity =  self.CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name =  self.le.classes_[j]
                        text = "{}".format(name)
                        print("Recognized: {} <{:.2f}>".format(name, proba * 100))
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (179, 0, 149), 4)
            else:
                for tracker, text in zip(trackers, texts):
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    cv2.rectangle(frame, (startX, startY), (endX, endY), (179, 0, 149), 4)
                    cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)

            self.fps.update()
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
