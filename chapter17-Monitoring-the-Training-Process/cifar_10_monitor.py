import matplotlib
matplotlib.use("agg")
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.nn.conv.minivggnet import MiniVGGnet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import cifar10
import tensorflow as tf
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from keras.callbacks import LearningRateScheduler

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required= True,help="path to output dirctory")
args = vars(ap.parse_args())

print("[INFO] process ID: {} ".format(os.getpid()))

print("[INFO] loading dataset...")
#replace fetch_mldata with 
((x_train,y_train),(x_test,y_test)) = cifar10.load_data()
x_train =x_train.astype("float")/255.0
x_test =x_test.astype("float")/255.0

#convert labels from int to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)
label_names =["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate =0.01, momentum = 0.9,nesterov=True)
model = MiniVGGnet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath,jsonPath)]

print("[INFO] training the model...")
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=40,verbose=1,callbacks=callbacks)

