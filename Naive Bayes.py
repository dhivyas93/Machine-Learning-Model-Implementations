# importing necessary libraries
import numpy as np
import pandas as pd
import random
import math
import operator
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# method to clean data
def cleanData(dataset):
    for col in dataset.columns:
        if(dataset.loc[:,col].dtype == 'object'):
            mylist = list(dataset.loc[:,col])
            mode = max(set(mylist), key=mylist.count)
            dataset.loc[:,col] = np.where(dataset.loc[:,col] == '?', mode, dataset.loc[:,col])
            dataset.loc[:,col].fillna(mode)
            dataset.loc[:,col], indexer = pd.factorize(dataset.loc[:,col])
        else:
            median = np.median(dataset.loc[:,col])
            dataset.loc[:,col].fillna(median)
    return dataset

# supporting methods to clean the data
def encodeData(data):
    le = LabelEncoder()
    le.fit(data)
    y = le.transform(data)
    return y

def encodeCategoricalVariables(dataset):
    for col in dataset.columns:
        data = dataset[col]
        if(dataset[col].dtype == 'object'):
            dataset[col] = encodeData(data)
    return dataset

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
        columnSize = len(train_dataset_split[i].columns)
        columns = train_dataset_split[i].columns
        ytrain = np.array(train_dataset_split[i].loc[:, columns[columnSize-1]])
        Xtrain = np.array(train_dataset_split[i].loc[:, columns[0:columnSize-1]])
        ytest = np.array(test_dataset_split[i].loc[:, columns[columnSize-1]])
        Xtest = np.array(test_dataset_split[i].loc[:, columns[0:columnSize-1]])
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        accuracy.append(1 - model.score(Xtest, ytest))
    plt.plot(range(len(train_dataset_split)), accuracy)
    plt.show()
    return np.mean(accuracy)


# Method to plot results of the k-fold cross validation for naive bayes model
def plot_test_accuracies(model, dataset):
    accuracies = []
    accuracy = KFoldCrossValidation(model, dataset)
    print("Average Error is : " , str(accuracy))
    return accuracy


# method implementing Naive Bayes model from Scratch
class NaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)] for i in separated])
        return self

    def _prob(self, x, mean, std):
        exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
        return np.log(exponent / (np.sqrt(2 * np.pi) * std))
    
    def predict_log_proba(self, X):
        return [[sum(self._prob(i, *s) for s, i in zip(summaries, x)) for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# method depicting the usage of the implemented K-Nearest Neighbours model
def main():
    # dataset = input dataset
    model = NaiveBayes()

	print("IonoSphere Dataset")
	accuracy_nb = plot_test_accuracies(model, dataset)
main()