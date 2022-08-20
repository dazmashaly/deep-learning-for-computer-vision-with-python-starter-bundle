from pyimagesearch.nn.conv import lenet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


#load the data 
print("[INFO] loading dataset...")
#replace fetch_mldata with 
dataset = datasets.fetch_openml('mnist_784')

#scale the raw pixel intesities to [0,1] range
X = dataset.data.astype("float") /255.0
y = dataset.target.astype(np.int8) 
X = X.values.reshape(70000,28, 28,1)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

#convert labels from int to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate =0.01)
model = lenet.Lenet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[INFO] training the model...")
H = model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=20,verbose=1)

print("[INFO] evaluating the network...")
preds = model.predict(x_test,batch_size=128)
print(classification_report(y_test.argmax(axis=1),preds.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,20),H.history["loss"],label="training loss")
plt.plot(np.arange(0,20),H.history["val_loss"],label="validation loss")
plt.plot(np.arange(0,20),H.history["accuracy"],label="training acc")
plt.plot(np.arange(0,20),H.history["val_accuracy"],label="validation acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()