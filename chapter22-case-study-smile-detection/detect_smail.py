from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse 
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
help="path to pre-trained smile detector CNN")
ap.add_argument("-v", "--video",
help="path to the (optional) video file")
args = vars(ap.parse_args())
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])
classLabels = ["sad","happy"]
while True:
    #grab the current frame
    (grabbed, frame) = camera.read()

    #if no frame was grabed the we reached the end
    if args.get("video") and not grabbed:
        break

    #resize the frame ,grayscale it , and clone it to draw on it later
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()
    #detect faces
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=8, minSize=(40, 40),flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in rects:
         # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=0)
        
        #classify
        print(model.predict(roi)[0])
        
        (notSmiling, smiling) = model.predict(roi, batch_size=32)[0]
        label = "Smiling" if  notSmiling !=1 else "Not Smiling"
        #we have the label now draw it along with the roi 
        cv2.putText(frameClone, label, (fX, fY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
    #display it
    cv2.imshow("Face", frameClone)
    #if the q key was pressed stop the loop
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break
camera.release()
cv2.destroyAllWindows()