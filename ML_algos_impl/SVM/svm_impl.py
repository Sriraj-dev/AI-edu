import numpy as np

class SVM:
    def __init__(self,lr = 0.001,lambda_ = 0.01,n_iter = 1000):
        self.n_iter = n_iter
        self.lr = lr
        self.lambda_ = lambda_
        self.W = None
        self.b = None

    def fit(self,X,y):
        m,n = X.shape
        self.W = np.zeros(n)
        self.b = 0
        y = np.where(y<=0,-1,1)
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                cond = y[idx] * (np.dot(self.W,x_i) - self.b) >= 1
                if cond:
                    db = 0
                    dw = 2 * self.lambda_ * self.W

                    self.W -= self.lr * dw
                else:
                    dw = 2 * self.lambda_ * self.W - np.dot(y[idx],x_i)
                    db = y[idx]

                    self.W -= self.lr * dw
                    self.b -= self.lr * db


    def predict(self,X):
        return np.sign(np.dot(self.W,X) - self.b)

