import numpy as np


class LinearRegressor:
    def __init__(self,lr = 0.01,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.W = np.random.randn(3,)
        self.b = 0

    def fit(self,X,y):
        self.W = np.random.randn(X.shape[1])
        self.b = 0
        m = X.shape[0]

        #loop  over n_iters
        for _ in range(self.n_iter):
            # error calculation
            err = np.dot(X,self.W) + self.b - y

            # gradient calculation
            gradient_w = np.dot(X.T,err) / m
            gradient_b = np.sum(err) / m

            # descent step
            self.W = self.W - self.lr * gradient_w
            self.b = self.b - self.lr * gradient_b

    def predict(self,X):
        return np.dot(X,self.W) + self.b



