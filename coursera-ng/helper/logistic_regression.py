
# coding: utf-8

# In[9]:

import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic('matplotlib inline')

import scipy.optimize as opt


# In[10]:

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[11]:

def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


# In[12]:

def gradient(theta, X, y):
    '''batch gradient'''
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


# In[13]:

def predict(X, theta):
    prob = sigmoid(X @ theta)
    return (prob >= 0.5).astype(int)


# In[14]:

def regularized_cost(theta, X, y, rate=1):
    regularized_term = (1 / (2 * len(X))) * np.power(theta, 2).sum()
    
    return cost(theta, X, y) + regularized_term


# In[15]:

def regularized_gradient(theta, X, y, rate=1):
    regularized_theta = (1 / len(X)) * theta
#     print(regularized_theta.shape)
#     regularized_therm = np.concatenate([np.array([0]), regularized_theta])
    
    return gradient(theta, X, y) + regularized_theta


# In[16]:

def logistic_regression(X, y, rate=1):
    theta = np.zeros(X.shape[1])
    
    res = opt.minimize(fun=regularized_cost,                       x0=theta,                       args=(X, y, rate),                       method='TNC',                       jac=regularized_gradient,                       options={'disp': True})
    
    final_theta = res.x
    
    return final_theta

