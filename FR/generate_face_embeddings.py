
from FR.insightface.deploy import face_model
from imutils import paths
import numpy as np
import pickle
import cv2
import os



class GenerateFaceEmbedding:

    def __init__(self):
        self.image_size = '112,112'
        self.model = os.path.sep.join(
            [str(os.getcwd()), "FR/models/model-y1-test2/model,0"])
           
        self.embedding_model_path = os.path.sep.join(
            [str(os.getcwd()), "FR/faceEmbeddingModels/embeddings.pickle"])
        self.threshold = 1.24
        self.det = 0

    def genFaceEmbedding(self, path):
        # Grab the paths to the input images in our datase
        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images(path))
        print(imagePaths)

        # Initialize the faces embedder
        embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

        # Initialize our lists of extracted facial embeddings and corresponding people names
        knownEmbeddings = []
        knownNames = []

        # Initialize the total number of faces processed
        total = 0

        # Loop over the imagePaths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            # load the image
            image = cv2.imread(imagePath)
            # convert face to RGB color
            nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nimg = np.transpose(nimg, (2, 0, 1))
            # Get the face embedding vector
            face_embedding = embedding_model.get_feature(nimg)

            # add the name of the person + corresponding face
            # embedding to their respective list
            knownNames.append(name)
            knownEmbeddings.append(face_embedding)
            total += 1

        print(total, " faces embedded")

        # save to output
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        f = open(self.embedding_model_path, "wb")
        f.write(pickle.dumps(data))
        f.close()
print("done embedding")