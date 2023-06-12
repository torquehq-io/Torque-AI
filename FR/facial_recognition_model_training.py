from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import argparse
import pickle

from FR.softmax import SoftMax

import os

class TrainFaceRecogModel:

    def __init__(self):
        # Load the face embeddings
        self.data = pickle.loads(open(os.path.sep.join(
            [str(os.getcwd()) ,"FR/faceEmbeddingModels/embeddings.pickle"]), "rb").read())

    def trainKerasModelForFaceRecognition(self):
        # Encode the labels
        le = LabelEncoder()
        labels = le.fit_transform(self.data["names"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()
        labels = one_hot_encoder.fit_transform(labels).toarray()

        embeddings = np.array(self.data["embeddings"])

        # Initialize Softmax training model arguments
        BATCH_SIZE = 8
        EPOCHS = 200
        input_shape = embeddings.shape[1]

        # Build sofmax classifier
        softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
        model = softmax.build()

        # Create KFold
        cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
        history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
            his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))


            history['acc'] += his.history['accuracy']
            history['val_acc'] += his.history['val_accuracy']
            history['loss'] += his.history['loss']
            history['val_loss'] += his.history['val_loss']

            print(f"[INFO] Accuracy = {history['acc']}")

        # write the face recognition model to output
        model.save(os.path.sep.join(
            [str(os.getcwd()), "FR/faceEmbeddingModels/my_model.h5"]))
        f = open(os.path.sep.join(
            [str(os.getcwd()) ,"FR/faceEmbeddingModels/le.pickle"]), "wb")
        f.write(pickle.dumps(le))
        f.close()
