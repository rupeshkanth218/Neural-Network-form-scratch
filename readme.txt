1. we need to feed input data to the network
2. The data flows from layer to layer until we have output
3. once we have output calculate error (scalar)
4. Finally we can adjust parameters(weight, bias) by subtracting 
    derivative of error with respect to parameter itself.
5. repeat this process

X--->LAYER---->Y

Forward Propagation


X--->LAYER1---->LAYER2----->LAYER3---->Y,error


Backward Propagation: To update the weights based on error

The goal is to minimize the error by changing the parameters in the n/w 

Gradient Descent

w<--- w-alpha*δE/δw

δE/δX= δE/δY   *   W.T
δE/δw= X.T   *  δE/δY
δE/δB= δE/δY
