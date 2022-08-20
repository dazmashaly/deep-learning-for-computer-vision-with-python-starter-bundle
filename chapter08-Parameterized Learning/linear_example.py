import numpy as np
import cv2
from imutils import paths

labels=["dog","cat","panda"]
np.random.seed(1)
w = np.random.randn(3,3072)
b=np.random.randn(3)
orig = cv2.imread("dogs_00322.jpg")
image = cv2.resize(orig,(32,32)).flatten()
scores = w.dot(image) + b
for (label,score) in zip(labels,scores):
    print("[INFO] {}: {:.2f}".format(label,score))
cv2.putText(orig,"Label: {}".format(labels[np.argmax(scores)]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
cv2.imshow("image",orig)
cv2.waitKey(0)    