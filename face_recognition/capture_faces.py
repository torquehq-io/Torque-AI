import cv2
from mtcnn import MTCNN
import imutils
from utils.insightface.src.common import face_preprocess
import datetime
import os
import numpy as np


class TrainingDataCollector:
    def __init__(self):
        self.detector = MTCNN()

    def collectImages(self, max_faces, name, path):
        video_cap = cv2.VideoCapture(0)
        # Setup some useful var
        faces = 0
        frames = 0
        max_bbox = np.zeros(4)
        # In case no gpu is there uncomment below line
        # video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while faces< max_faces:
            # Uncomment below line in case no GPU is there
            # video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ret, frame = video_cap.read()
            if ret == False:
                print("No Frame is there!!")
                break
            else:
                frame = imutils.resize(frame, width=1280, height=640)
                frames += 1
                dtString = str(datetime.datetime.now().microsecond)

                text1 = f"Capturing {name}'s images for training the model"
                text2 = f"Images Capured = {faces}"
                cv2.putText(frame, text1,
                            (int(0.3 * frame.shape[1]), int(0.05 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 0, 0), 2)
                cv2.putText(frame, text2,
                            (int(0.4 * frame.shape[1]), int(0.1 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 0, 0), 2)

                # Detect the face using MTCNN detector
                results = self.detector.detect_faces(frame)
                #print(results)

                # Print the bounding box on face
                if len(results) != 0:
                    # Get only the biggest face
                    max_area = 0
                    for result in results:
                        bbox = result["box"]
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        keypoints = result["keypoints"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > max_area:
                            max_bbox = bbox
                            landmarks = keypoints
                            max_area = area

                    max_bbox = max_bbox[0:4]

                    # get each of 3 frames
                    if frames % 3 == 0:
                        # convert to face_preprocess.preprocess input
                        landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                              landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                              landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                              landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                        landmarks = landmarks.reshape((2, 5)).T
                        nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')
                        if os.path.exists(path):
                            cv2.imwrite(os.path.join(path, f"{dtString}.jpg"), nimg)
                        else:
                            os.mkdir(path)
                            cv2.imwrite(os.path.join(path, f"{dtString}.jpg"), nimg)

                        cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                        text = f"Captured image {faces+1}"
                        cv2.putText(frame, text,
                                    (max_bbox[0]-2, max_bbox[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (255, 0, 0), 2)

                        print("[INFO] {} Image Captured".format(faces + 1))
                        faces += 1
                cv2.imshow("Face detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_cap.release()
        cv2.destroyAllWindows()
