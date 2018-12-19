# importing necessary libraries
import pandas as pd 
import numpy as np
import scipy
import matplotlib.pyplot as plt

# method to normalize the numerical features using mean and standard deviation method
def feature_normalization(X):
    '''
    In this function we are going to normalize all the features. 
    
    - for each feature, calculate its mean
    - substract the mean from their respective feature
    
    - for each feature, calculate its standard deviation
    - divide each feature by its standard deviation
    
    '''
  
    n = X.shape[1]
    X_norm = X
    X_norm = pd.DataFrame(X_norm)
    mu = np.zeros(n)
    sigma = np.zeros(n)
    
    for i in range(n):
        mu[i] = np.mean(X_norm.loc[:,i])
        sigma[i] = np.std(X_norm.loc[:, i])
        X_norm.loc[:, i] = (X_norm.loc[:,i] - mu[i])/sigma[i]
        
    return X_norm, mu, sigma



# method to find the cost function for the model
def cost_function(X, y, theta):
    '''
    In this function the cost function is implemented
    using the basic cost function formula
 
    '''
  
    m = y.size
    cost = 0
    
    cost = np.sum((np.dot(X, theta) - y)**2) / m
    
    return cost


# method to implement Linear Regression
def linearRegression_ols(X, y):
    '''
    Implement the closed-form (or ordinary least squares) solution for
    linear regression. The result is saved in the variable 'theta'.
    
    '''
    theta = np.zeros((X.shape[1]))
    transposedX = np.transpose(X)
    
    comp1 = np.linalg.inv(np.dot(transposedX, X))
    comp2 = np.dot(transposedX, y)
    
    theta = np.dot(comp1, comp2)
    
    return theta


# method depicting usage of the Linear regression model implemented
def main():
	
	# X denotes a training dataset without the target variable
	# y denotes the training target variable data

	X = np.c_[np.ones(m), X] # Add intercept term to X

	theta = linearRegression_ols(X, y)