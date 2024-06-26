import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=4)

# python model prediciton
from algo import LogisticRegression
from sklearn import metrics

model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("Python model Accuracy: " + str(metrics.accuracy_score(y_test,predictions)))

# sklearn impl
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

print("sklearn model accuracy: " + str(metrics.accuracy_score(y_test,predictions)))


