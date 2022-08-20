from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import shallownet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizer_v1 import sgd
from keras.optimizers import sgd_experimental
from keras.datasets import cifar10
import tensorflow as tf
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import argparse



#download the 170MB cifar10 dataset
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

#initialize the optimizer and the model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
model = shallownet.ShallowNet.build(width=32,height=32,classes=10,depth=3)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

#train the network
print("[INFO] training the network...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=32,epochs=45,verbose=1)

#evaluate the model
print("[INFO] evaluating the model...")
preds = model.predict(x_test,batch_size=32)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]))

#plot
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,45),H.history["loss"],label="training loss")
plt.plot(np.arange(0,45),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,45),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,45),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()