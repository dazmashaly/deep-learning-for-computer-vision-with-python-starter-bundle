from pyimagesearch.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to the input dataset")
ap.add_argument("-m","--model",required=True,help="path to input model")
args= vars(ap.parse_args())
classLabels = ["sad","happy"]

print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

sp = SimplePreprocessor(28, 28)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)

data = data.astype("float") / 255.0
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
     # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (200, 200))
    cv2.putText(image, "{}".format(classLabels[preds[i]]),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)