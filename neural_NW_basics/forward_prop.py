import numpy as np
# import tensorflow as tf
from activation_functions import sigmoid


#Blueprint of a layer
class Layer:
    def __init__(self,units,W,b,name=""):
        self.W = W
        self.b = b
        self.units = units
        self.name = name


#Dense layer which executes the operations of a particular layer
def dense(a_in,W,b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for i in range(units):
        w = W[:,i]
        b_i = b[i]
        z = np.dot(w,a_in) + b_i
        a_out[i] = sigmoid(z)
    return a_out


#Sequential neural network which executes the forward propogation for a particular input
def sequential(a_in,layers):
    n = len(layers)

    #Moving layer by layer to calculate the output of neural network
    for i in range(n):
        a_out = dense(a_in,layers[i].W,layers[i].b)
        a_in = a_out
    
    return a_in

#predicting the output of neural network for given set of training examples
def predict(X,layers):
    #m = no. of training examples
    m = X.shape[0]
    p_out = np.zeros((m,1))
    for i in range(m):
        p_out[i] = sequential(X[i],layers)
    return p_out

#normalising the input using tensorflow
# def normalise():
#     norm_l = tf.keras.layers.Normalization(axis=-1)
#     norm_l.adapt(X)  # learns mean, variance
#     Xn = norm_l(X)



#Create Layers with weights
layers = []
#Layer1
layers.append(Layer(3,np.array([[-8.93,  0.29, 12.9 ], [-0.1,  -7.32, 10.81]]),np.array([-9.82, -9.28,  0.96]),"layer1"))
#layer2
layers.append(Layer(1,np.array([[-31.18], [-27.59], [-32.56]]),np.array([15.41]),"layer2"))

#input to neural network after normalising:
X_train = np.array([[-0.47,0.42],[-0.47,3.16]])

predictions = predict(X_train,layers)
