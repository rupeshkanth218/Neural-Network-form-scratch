import numpy as np
from network import Network
from activation import ActivationLayer, tanh, tanh_prime
from fc_layer import FCLayer
from loss_function import mse, mse_prime

# xor problem data
X=np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])

Y=np.array([[[0]], [[1]], [[1]], [[0]]])

# lets build network
net=Network()
net.add_layer(FCLayer(2,3))
net.add_layer(ActivationLayer(tanh,tanh_prime))
net.add_layer(FCLayer(3,1))
net.add_layer(ActivationLayer(tanh,tanh_prime))

net.use_loss(mse,mse_prime)
net.fit(X,Y,epochs=1100,learning_rate=0.1)


# test

out=net.predict(X)
print(out)

#[array([[0.00078036]]), array([[0.97830563]]), array([[0.97821458]]), array([[-0.00142725]])]
#[array([[-0.00317202]]), array([[0.98540289]]), array([[0.98537629]]), array([[-0.02898182]])]