import numpy as np
import statistics as st

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X,y):
        self.X = X
        self.y = y

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self,x):
        dist = np.sqrt(np.sum((self.X - x) ** 2 , axis=1))
        k_indices = np.argsort(dist)[:self.k]
        return st.mode(self.y[k_indices])

        