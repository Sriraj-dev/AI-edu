# In summary, a perceptron is a specific type of neuron that typically uses a step activation 
# function and is limited to learning linear decision boundaries, while neurons in general are 
# more flexible and versatile, capable of learning complex patterns and relationships in data.

import numpy as np

class Perceptron:
    def __init__(self,lr = 0.01,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.activation_func = self._unit_step_func
        self.W = None
        self.b = None

    def _sigmoid(self,x):
        return 1 / (1 +np.exp(-x))
    
    def _unit_step_func(x):
        return np.where(x>=0, 1, 0)
    
    def fit(self,X,y):
        m,n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iter):
            # prediction
            pred = self.activation_func(np.dot(X,self.W) + self.b)

            # gradient
            gradient_w = np.dot(X.T,(pred - y)) / m
            gradient_b = np.sum(pred - y) / m

            #descent
            self.W -= self.lr * gradient_w
            self.b -= self.lr * gradient_b

    def predict(self,X):
        return self.activation_func((np.dot(X,self.W) + self.b))

