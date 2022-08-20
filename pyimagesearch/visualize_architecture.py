from re import L
from pyimagesearch.nn.conv.lenet import Lenet
from keras.utils.all_utils import plot_model

#initziale the networl
model = Lenet.build(28,28,1,10)
plot_model(model,to_file="lenet.png",show_shapes=True)