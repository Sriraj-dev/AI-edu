import numpy as np


class LogisticRegression:
    def __init__(self,lr = 0.01,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.W = None
        self.b = None

    def _sigmoid(self,x):
        return 1 / (1 +np.exp(-x))
    
    def fit(self,X,y):
        m,n = X.shape
        self.W = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iter):
            # prediction
            pred = self._sigmoid(np.dot(X,self.W) + self.b)

            # gradient
            gradient_w = np.dot(X.T,(pred - y)) / m
            gradient_b = np.sum(pred - y) / m

            #descent
            self.W -= self.lr * gradient_w
            self.b -= self.lr * gradient_b

    def predict(self,X):
        return [1 if i> 0.5 else 0 for i in self._sigmoid((np.dot(X,self.W) + self.b))]

