from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing import simplepreprocessor
from pyimagesearch.datasets import simpledatasetloader
from imutils import paths
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import argparse
# construct the argument parse and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input dataset")
ap.add_argument("-k","--neighbors",type=int,default=1,help="# of nearest neighbors for calssfication")
ap.add_argument("-j","--jobs",type=int,default=-1,help="# of jobs for knn distance -1 use all")
args = vars(ap.parse_args())

#grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

#initialize the image preprocessor, load the dataset, reshape the matrix
sp = simplepreprocessor.SimplePreprocessor(32,32)
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])
(data,labels) = sdl.load(imagePaths,verbose=500)
data = data.reshape((data.shape[0],3072))


#show some information on memory consumption
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024*1000.0)))

# encode the labels as integers (convert labels form string to uniqe int per class)
le = LabelEncoder()
labels = le.fit_transform(labels)

#partiton the data into training and testing splits 75% to 25%
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=42)
print(trainX.shape)
print(trainY.shape)

#train and evaluate the model
print("[INFO] evaluating knn classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))
print(confusion_matrix(testY,model.predict(testX)))
print(accuracy_score(testY,model.predict(testX)))




