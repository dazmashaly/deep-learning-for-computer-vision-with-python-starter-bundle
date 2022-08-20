from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizer_v1 import sgd
from keras.optimizers import sgd_experimental
from sklearn import datasets
import tensorflow as tf
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import argparse



#construct the args and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to the output loss/acc plot")
args = vars(ap.parse_args())

#download the 55MB MNIST dataset
print("[INFO] loading dataset...")
#replace fetch_mldata with 
dataset = fetch_openml('mnist_784')
print(dataset.target)
#scale the raw pixel intesities to [0,1] range
X = dataset.data.astype("float") /255.0
y = dataset.target.astype(np.int8) 
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#convert labels from int to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

#define the network architecture 
model = Sequential()
model.add(Dense(256, input_shape=(784,),activation="sigmoid"))
model.add(Dense(128,activation="sigmoid"))
model.add(Dense(10,activation="softmax"))

#train the model
print("[INFO] training the model...")
sgD = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss="categorical_crossentropy",optimizer=sgD,metrics=["accuracy"])
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,batch_size=128)
print("[INFO] evaluating the model...")
preds = model.predict(x_test,batch_size=128)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

#plot the loss
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






















