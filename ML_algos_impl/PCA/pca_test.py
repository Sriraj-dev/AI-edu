import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = datasets.load_iris()
X,y = data.data,data.target

train_X,test_X,train_y,test_y = train_test_split(X,y,random_state=1)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(train_X)


pca = PCA(n_components=2)

pca.fit(scaled_X)

pcax = pca.transform(scaled_X)

plt.scatter(pcax[:,0],pcax[:,1] , c = train_y,cmap='viridis')

plt.show()
