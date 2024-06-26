
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=1000,n_features=1,n_targets=1,noise=4,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)

def plot_line():
    y_pred_line = model.predict(X)
    plt.figure(figsize=(8,6))
    plt.scatter(X_train,y_train,color = 'blue',s=20)
    plt.scatter(X_test,y_test,color = 'green',s=20)
    plt.plot(X,y_pred_line,color = 'black')

# print(X_test.shape)
# print(y_train.shape)

from LR_Impl import LinearRegressor
from sklearn import metrics

## predicting from python model:
model = LinearRegressor(lr = 0.5,n_iter= 1000)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Python Model Accuracy:" + str(metrics.mean_squared_error(y_true=y_test,y_pred=predictions)))

## plotting the graph
plot_line()

from sklearn import linear_model

## predicting from sklearn implementation
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
predictions  =model.predict(X_test)
print(model.coef_)

print("Sklearn Model Accuracy:" + str(metrics.mean_squared_error(y_true=y_test,y_pred=predictions)))


## showing the plot (from python model)
plt.show()


