# importing necessary libraries
import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# method to split data into train and test sets
def get_train_test_split(dataset, split = 0.8):
    np.random.seed(456)
    msk1 = np.random.rand(len(dataset)) <= split
    train = pd.DataFrame(dataset[msk1])
    test = pd.DataFrame(dataset[~msk1])
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    if 'index' in train.columns:
        train = train.drop(columns=['index'])
    if 'index' in test.columns:
        test = test.drop(columns=['index'])
    return train,test

# method to split data into given number of folds and split each fold into train and test data
def cross_validation_split(dataset, folds=5, split = 0.8):
    train_dataset_split = {}
    test_dataset_split = {}
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        random.seed(123)
        msk = random.sample(range(len(dataset)), fold_size)
        fold = pd.DataFrame(dataset.loc[msk, :])
        dataset = dataset.drop(fold)
        dataset.reset_index(inplace=True)
        if 'index' in dataset.columns:
            dataset = dataset.drop(columns=['index'])
        train, test = get_train_test_split(fold)
        train_dataset_split[i] = train
        test_dataset_split[i] = test
    return train_dataset_split, test_dataset_split

# method implementing K-Fold cross validation
def KFoldCrossValidation(model, dataset):
    accuracy = []
    train_dataset_split, test_dataset_split = cross_validation_split(dataset)
    for i in range(len(train_dataset_split)):
        model.fit(train_dataset_split[i])
        pred = model.predict(test_dataset_split[i])
        accuracy.append(model.model_score())
    return np.mean(accuracy)

# method to plot the results of K-Fold Cross Validation of a model
def plot_test_accuracies(model, dataset, k_list):
    accuracies = []
    for k in k_list:
        accuracy = KFoldCrossValidation(model, dataset)
        print("Average Error for k =", str(k), " is : " , str(1-accuracy))
        accuracies.append(1-accuracy)
    plt.plot(k_list, accuracies)
    plt.xlabel("k-value")
    plt.ylabel("Error on test data")
    plt.show()

# method depicting the usage of the implemented K-Fold Cross validation method
def main():
	k_list = [1,2,3,4,5]
	# dataset = input dataset
	plot_test_accuracies(dataset, k_list)
main()