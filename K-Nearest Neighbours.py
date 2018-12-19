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


# A Class implementing K-Nearest Neighbours model from scratch
class KNearestNeighbors():
    
    def __init__(self,k,distance):
        self.k=k
        self.distance=distance
    
    # methof to fit the training data X
    def fit(self,X):
        self.trainingSet=X

    # method to calculate euclidean distance function between two instances
    def euclidean_distance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += pow((float(instance1[x]) - float(instance2[x])), 2)
        return math.sqrt(distance)

    # method to calculate manhattan distance function between two instances
    def manhattan_distance(self, instance1, instance2, length):
        distance = 0
        for x in range(length):
            distance += abs(float(instance1[x]) - float(instance2[x]))
        return distance

    # method to obtain the nearest neighbours for a test instance
    def getNeighbors(self, testInstance):
        distances = []
        length = len(testInstance)-1
        for x in range(len(self.trainingSet)):
            if(self.distance == "Euclidean"):
                dist = self.euclidean_distance(testInstance, self.trainingSet.loc[x,:], length)
            else:
                dist = self.manhattan_distance(testInstance, self.trainingSet.loc[x,:], length)
            distances.append((self.trainingSet.loc[x,:], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(self.k):
            neighbors.append(list(distances[x][0]))
        return neighbors


    # method to get the output based on the neighbours
    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]

    # method to evaluate model score or accuracy
    def model_score(self):
        y_test = self.testSet.loc[:,len(self.testSet.columns)-1]
        return np.mean(self.predictions == y_test)  

    # method to predict the target variable for a given test set
    def predict(self, testSet):
        self.testSet = testSet
        predictions=[]
        for x in range(len(testSet)):
            neighbors = self.getNeighbors(self.testSet.loc[x,:])
            result = self.getResponse(neighbors)
            predictions.append(result)
        self.predictions = predictions
        return self.predictions


# method depicting the usage of the implemented K-Nearest Neighbours model
def main():
    # dataset = input dataset
    k_list = [1,2,3,4,5]
    print("-----------EUCLIDEAN-----------")
    test_accuracies(dataset, k_list, "euclidean")
    print("-----------MANHATTAN-----------")
    test_accuracies(dataset, k_list, "manhattan")
main()