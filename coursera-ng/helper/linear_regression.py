
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:

def cost(theta, X, y):
    """
    X: R(m*n)
    y: R(m)
    theta: R(n)
    """
    m, n = X.shape
    
    loss = (X @ theta) - y
    square_sum = loss.T @ loss
    cost = square_sum / (2 * m)
    
    return cost


# In[6]:

def gradient(theta, X, y):
    m, n = X.shape
    
    loss = X.T @ (X @ theta - y)
    return loss / m


# In[ ]:

def batch_gradient_decent(theta, X, y, maxitr, alpha=0.01):
    _cost = [cost(theta, X, y)]
    _theta = theta.copy()
    
    for _ in range(maxitr):
        _theta = _theta - alpha * gradient(_theta, X, y)
        _cost.append(cost(_theta, X, y))
        
    return _theta, _cost

