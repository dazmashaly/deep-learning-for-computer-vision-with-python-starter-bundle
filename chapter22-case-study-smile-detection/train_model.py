
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array
from pyimagesearch.nn.conv.minivggnet import MiniVGGnet
from keras.utils import np_utils
from pyimagesearch.nn.conv.lenet import Lenet
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required= True,help="path to output dirctory")
ap.add_argument("-m","--model",required= True,help="path to output dirctory")
args = vars(ap.parse_args())

data = []
labels =[]

for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image,width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smailing" if label == "positives" else "not smiling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals
class_weight = {0:3.,1:1}

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)
print("[INFO] compiling model...")
model = MiniVGGnet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss="binary_crossentropy", optimizer="adam",
metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64,class_weight=class_weight, epochs=16, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] serilizeing network...")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,16),H.history["loss"],label="training loss")
plt.plot(np.arange(0,16),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,16),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,16),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()



