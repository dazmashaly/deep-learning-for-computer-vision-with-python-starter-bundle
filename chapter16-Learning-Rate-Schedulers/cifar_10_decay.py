import matplotlib
matplotlib.use("agg")

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
from keras.callbacks import LearningRateScheduler

def step_decay(epoch):
    #initialze the base learning rate ,drop factor and epochs
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 5

    #compute the learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    return alpha

ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to save")
args = vars(ap.parse_args())

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

# define the list of callbacks to be passed to the model
callbacks = [LearningRateScheduler(step_decay)]

print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate =0.01, momentum = 0.9,nesterov=True)
model = MiniVGGnet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training the model...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=40,batch_size=40,verbose=1,callbacks=callbacks)
print("[INFO] evaluating the model...")
preds = model.predict(x_test,batch_size=64)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=label_names))
