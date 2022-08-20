from numpy import var
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivggnet import MiniVGGnet
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w","--weights",required=True,help="path to weights dirctory")
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


print("[INFO] compiling model...")
opt = tf.keras.optimizers.SGD(learning_rate =0.01,decay = 0.01/40, momentum = 0.9,nesterov=True)
model = MiniVGGnet.build(width=32,height=32,depth=3,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

#construct callback to only save the best model 
fname = os.path.sep.join([args["weights"],"weights -{epoch:03d}-{val_loss:.4f}.hdf5"])
cheakpoint = ModelCheckpoint(args["weights"],monitor="val_loss",save_best_only=True,verbose=1)
callbacks = [cheakpoint]
print("[INFO] training the model...")
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=40,batch_size=40,verbose=1,callbacks=callbacks)