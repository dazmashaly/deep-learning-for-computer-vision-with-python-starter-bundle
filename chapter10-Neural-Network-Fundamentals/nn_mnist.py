from matplotlib.pyplot import axis
from pyimagesearch.nn import nenet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets
 
# load the mnist dataset and apply min/max scaling to scale the 
#pixel intensity to the range [0,1]
print("[INFO] loading mnist (sample) dataset...")
digits = datasets.load_digits()
data = digits.data.astype("float")
data = (data -data.min())/(data.max() - data.min())
print("[INFO] samples: {}, dim: {}".format(data.shape[0],data.shape[1]))
x_train,x_test,y_train,y_test = train_test_split(data,digits.target,test_size=0.25,random_state=42)
#convert the labels from int to vectors
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)
print("[INFO] training network...")
nn = nenet.NeuralNetwork([x_train.shape[1],32,16,10])
print("[INFO] {}".format(nn))
nn.fit(x_train,y_train,epochs=1000)
print("[INFO] evaluationg model...")
preds = nn.predict(x_test)
preds= preds.argmax(axis=1)
print(classification_report(y_test.argmax(axis=1),preds))
print(confusion_matrix(y_test.argmax(axis=1),preds))
print(accuracy_score(y_test.argmax(axis=1),preds))