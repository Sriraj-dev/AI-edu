

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def entropy(p):
    if p==0 or p==1:
        return 0
    else:
        return -p * np.log2(p) - (1-p) * np.log2(1-p)
    

def weighted_entropy(X,y,left_indices,right_indices):

    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)

    return w_left * entropy(p_left) + w_right * entropy(p_right)


def information_gain(X,y,left_indices,right_indices):
    n_entropy = entropy(sum(y)/len(y))
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return n_entropy - w_entropy

    

#split the indices based on a feature:
def split_indices(X, index_feature):
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    
    return left_indices,right_indices



##Split the data on each feature and compare which is giving me higher information gain

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])




for i in range(X_train.shape[1]):
    left_indices , right_indices = split_indices(X_train,i)
    i_gain = information_gain(X_train,y_train,left_indices,right_indices)
    print(f"If we split the node suing feature {i} , i_gain would be = {i_gain}")

## This process is done recursively to build the entire tree.

