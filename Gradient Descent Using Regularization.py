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
def cost_function_regularization(X, y, theta):
    '''
    In this function the cost function is implemented
    using the basic cost function formula
 
    '''

    Lambda = 10
    
    m = y.size
    cost = 0
    
    cost = np.sum((np.dot(X, theta) - y)**2)/m #+ Lambda*(np.sum(theta**2))
    
    return cost


# method to implement the Gradient Descent Regression using Regularization
def gradient_descent_regularization(X, y, theta, learning_rate, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    p = 2

    Lambda = 10
    for i in range(0, num_iters):
        '''
        Implement gradient descent for a single gradient step on the parameter 
        vector theta. Save the result of each iteration on J_history.
        '''
        gradient = (learning_rate/m) * 2 * np.dot((np.dot(X, theta) - y) , X)
        
        theta = theta*(1- ((learning_rate*Lambda)/m)) - gradient
    
        J_history[i] = cost_function_regularization(X, y, theta)

    return theta, J_history


# method depicting usage of the Gradient Descent regression using regularization model implemented
def main():
    
    # X denotes a training dataset without the target variable
    # y denotes the training target variable data

    # Normalize features
    print('Normalizing Features ...')

    X, mu, sigma = feature_normalization_regularization(X)
    X = np.c_[np.ones(m), X]  # Add a column of ones to X
    # Now we proceed with Gradient Descent
    X = pd.DataFrame(X)
    print('Running gradient descent ...')

    # Choose some alpha value
    alpha = 0.03
    num_iters = 400

    # Initialize theta and execute gradient descent
    # theta = np.zeros(3)
    theta = np.zeros((X.shape[1]))
        
    theta, J_history = gradient_descent_regularization(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.figure()
    plt.plot(np.arange(J_history.size), J_history)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    # Display gradient descent's result

    print('Theta computed from gradient descent : \n{}'.format(theta))

main()