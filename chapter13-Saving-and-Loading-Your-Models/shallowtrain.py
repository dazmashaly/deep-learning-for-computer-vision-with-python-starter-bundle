from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import imagetoarraypreprocessor
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from pyimagesearch.nn.conv import shallownet
import tensorflow as tf
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to the input dataset")
ap.add_argument("-m","--model",required=True,help="path to input model")
args= vars(ap.parse_args())

print("[INFO] loding dataset... ")
imagePaths = list(paths.list_images(args["dataset"]))

sp = simplepreprocessor.SimplePreprocessor(32,32)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

sdl =simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp,iap])
data,labels =sdl.load(imagePaths,verbose=500)
data = data.astype("float")/255.0

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.25,random_state=42)

#convert labels from intgers to vector
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

#initialize the optimizer and the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate=0.005)
model = shallownet.ShallowNet.build(width=32,height=32,classes=3,depth=3)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training the model...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=100,verbose=1)

#saving the model
print("[INFO] serializing network...")
model.save(args["model"])

print("[INFO] evaluating the model...")
preds = model.predict(x_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=["cat","dog","panda"]))

#plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100),H.history["loss"],label="training loss")
plt.plot(np.arange(0,100),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,100),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,100),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()