import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X,y = datasets.make_classification(n_samples=500,n_features=10,n_classes=4,n_informative=8,random_state=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

print(X_train.shape)
print(X_train[0])
print(y_train.shape)
print(y_train[0])


#You can read the conept of decision tree from the jupyter notebook created in decision_trees.
#Single Decision Tree classifier:

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


model = DecisionTreeClassifier(random_state = 1,criterion='gini')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Single Decision Tree Regressor accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))

# Tree ensmeble - RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Random forest accuracy: " + str(metrics.accuracy_score(y_test,y_pred)))

#Tree ensemble - XGBoost
from XGBoost import XGBClassifier

model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("XGBoost accuracy : " + str(metrics.accuracy_score(y_test,y_pred)))


