import imp
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
import argparse

# construct the arguments and parse them 
ap= argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
args = vars(ap.parse_args())

#grab the list of image paths
print("[INFO] loading images...")
imagepaths = list(paths.list_images(args["dataset"]))

#initalize the image processor and load the data
sp = simplepreprocessor.SimplePreprocessor(32,32)
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagepaths,verbose=500)
data = data.reshape((data.shape[0]),3072)

#encode the labels as intger
le = LabelEncoder()
labels =le.fit_transform(labels)

#split the data
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.25,random_state=5)

#loop over the set of regulaizers
for r in  (None,"l1","l2"):
    #train a sgd classifier using a sotmax loss function and the specified 
    #regularzation function for 10 epochs
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log",penalty=r,max_iter=10,learning_rate="constant",eta0=0.01,random_state=42)
    model.fit(x_train,y_train)

    #evaluate 
    acc=model.score(x_test,y_test)
    print("[INFO] '{}' penalty accuracy {:.2f}%".format(r,acc*100))