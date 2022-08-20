from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np 
import argparse

def sigmoid_activation(x):
    # compute the sigmoid activation value for a given input
    return 1.0 / ( 1 + np.exp(-x))

def predict(x,w):
    # take the dot product of the feature and the weight matrix
    preds = sigmoid_activation(x.dot(w))
    # apply a step function to threshold the outputs to binary class labels
    preds[preds <= 0.5] = 0
    preds[preds>0.5] = 1
    return preds
def next_batch(x,y,batchSize):
    #loob over the dataset "x" in mini-batches ,yielding a tuple of batched data and labels
    for i in np.arange(0,x.shape[0],batchSize):
        #cut the data into batches and thier label
        yield(x[i:i + batchSize],y[i:i+batchSize])

#parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e","--epochs", type=float, default= 100, help= "# of epochs")
ap.add_argument("-a","--alpha", type= float, default= 0.01 , help= "learning rate")
ap.add_argument("-b","--batch-size",type=int,default=32,help="size of sgd mini batches")
args = vars(ap.parse_args())

#generate data 
(x,y) = make_blobs(n_samples= 1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)
y=y.reshape((y.shape[0],1))

# add a column of ones to the featuer matrix to treat the bias as a trainble paramter
x = np.c_[x,np.ones((x.shape[0]))]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=42)

# initialize our weight matrix and list of losses
print("[INFO] training...")
w = np.random.randn(x.shape[1],1)
losses = []

#loop over epochs
for epoch in np.arange(0,args["epochs"]):
    epoch_loss =[]

    #loop over the data in batches
    for(batchx,batchy) in next_batch(x,y,args["batch_size"]):
        #take the dot product between the current batch and the weight and pass the value to activation 
        preds = sigmoid_activation(batchx.dot(w))

        #determine the error
        error = preds - batchy
        epoch_loss.append(np.sum(error ** 2))

        #compute the gradient descent ,update the weight
        gradient= batchx.T.dot(error)
        w+= -args["alpha"] * gradient

    #update loss history taking the average of the batches in epochs
    loss =np.average(epoch_loss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch ==0 or (epoch +1) % 5 == 0:
        print("[INFO] epoch {} , loss = {:.7f}".format(int(epoch+1),loss))

# evaluate our model 
print("[INFO] evaluating...")
preds = predict(x_test,w)
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))
print(accuracy_score(y_test,preds))

# plot the data
plt.style.use("ggplot")
plt.figure()
plt.title("data")
plt.scatter(x_test[:, 0],x_test[:, 1],marker="o", c=y_test,s=30)

#plot the loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,args["epochs"]),losses)
plt.title("training loss")
plt.xlabel("epoch #")
plt.ylabel("loss")
plt.show()



