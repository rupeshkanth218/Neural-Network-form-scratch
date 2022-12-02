import numpy as np
from network import Network
from activation import ActivationLayer, tanh, tanh_prime
from fc_layer import FCLayer
from loss_function import mse, mse_prime

from keras.datasets import mnist
from keras.utils import np_utils

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Dataset loaded.")
# total samples 60_000

# preprocessing 
# reshape,  normalize the data
x_train=x_train.reshape(x_train.shape[0],1,28*28)
x_train=x_train.astype('float32')
x_train /= 255

y_train=np_utils.to_categorical(y_train)

x_test=x_test.reshape(x_test.shape[0],1,28*28)
x_test=x_test.astype('float32')
x_test /= 255

y_test=np_utils.to_categorical(y_test)

# create network
#1. FC layer 28*28,100
#2. activation layer
#2 Fc layer 100 , 50
#3 activation

# fc layer 50 10
# activatio
# which loss function to use
# train

net = Network()
net.add_layer(FCLayer(784, 100))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(100, 50))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.add_layer(FCLayer(50, 10))
net.add_layer(ActivationLayer(tanh, tanh_prime))
net.use_loss(mse, mse_prime)
print("Training started...")
net.fit(x_train[0:3000], y_train[:3000], epochs = 70, learning_rate = 0.1)
print("Training complete ")
output=net.predict(x_test[:5])
output=np.abs(np.rint(output))
for y_pred,y_true in zip(output,y_test[:5]):
    print(f"Prediction ={y_pred}, true ={y_true}")