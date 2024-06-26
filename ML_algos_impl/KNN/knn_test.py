import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X,y = iris.data,iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=1)

# print(X_train.shape)
# print(X_train[:5])
# print(y_train.shape)
# print(y_train[:5])

# plt.figure()
# plt.scatter(X[:,0] , X[:,1], c = y, cmap = 'magma',edgecolor = 'k', s = 20)
# plt.show()

##### Testing Our Model ######

from knn_algo import KNN

model = KNN(k=3)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
acc *= 100
print("Python Model Accuracy: " + str(acc))

##### Using Scikit learn #######
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Sklearn Accuracy:" + str(metrics.accuracy_score(y_test,predictions)))
