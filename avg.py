import numpy as np 
import pandas as pd 
import sys
import math
from collections import Counter

def predict(weight, bias, sample):
    score = np.sum(np.dot(weight, sample)) + bias #* len(sample)
    if score >= 0:
        return 1
    else:
        return 0

def perceptron(X, Y, max_it=2):
    bias = 0.0
    old_weights = [0.0] * len(X[0])
    weight = [0.0] * len(X[0])

    nT_rev = 1 / float(X.shape[0] * max_it)
    step = float(X.shape[0] * max_it)

    for it in range(max_it):
        for index in range(X.shape[0]):
            err = Y[index] - predict(weight, bias, X[index])
            if err != 0:
                bias += err
                bias *= step * nT_rev
                weight = np.add(weight, np.multiply(X[index], err))
                old_weights = np.add(old_weights, np.multiply(X[index], err * step * nT_rev))               
            step -= 1

    return old_weights, bias


def zero_one_loss(pred, actual):
    n = len(actual)

    if len(pred) != n:
        print 'something wrong'
        return 'null'
        
    loss = 0
    for pair in zip(pred, actual):
        if pair[0] != pair[1]:
            loss += 1
    return loss / float(n)

def main():

    """
    argv[0]: program name
    argv[1]: training file
    argv[2]: test file
    argv[3]: max_iteration
    Fetaures: Consider the first 14 discrete attributes in yelp_cat.csv for X.
    Class label: Consider the attribute goodForGroups as the class label, i.e., Y = {0, 1}.
    """


    # sanity checks
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print 'wrong number of arguments'
        return
    if len(sys.argv) == 4:
        try:
            max_iteration = int(sys.argv[3])
        except ValueError:
            print 'maximum iteration is not an integer'
            return
        if max_iteration < 0:
            print 'max_iteration should be non negative'
            return
    else:
        max_iteration = 2

    # import data
    try:
        raw_train_X = pd.read_csv(sys.argv[1], sep=',', quotechar='"', header=0, usecols=range(14), engine='python') 
        raw_train_Y = pd.read_csv(sys.argv[1], sep=',', quotechar='"', header=0, usecols=['goodForGroups'], engine='python')
    except IOError: 
        print 'training file not found'
        return
    try:
        raw_test_X = pd.read_csv(sys.argv[2], sep=',', quotechar='"', header=0, usecols=range(14), engine='python')
        raw_test_Y = pd.read_csv(sys.argv[2], sep=',', quotechar='"', header=0, usecols=['goodForGroups'], engine='python') 
    except IOError: 
        print 'testing file not found'
        return

    # length of train
    train_len = len(raw_train_X)

    # concate train and test
    inter = pd.concat([raw_train_X, raw_test_X])

    # transfrom to binary features
    inter_bi = pd.get_dummies(inter, columns=inter.columns.values)

    # split train and test
    train_X = inter_bi[:train_len]
    test_X = inter_bi[train_len:]

    # get test data and training data
    train_Y = [ x[-1] for x in raw_train_Y.as_matrix() ]
    test_Y = [ x[-1] for x in raw_test_Y.as_matrix() ]
    train_vals = train_X.as_matrix()
    test_vals = test_X.as_matrix()
    
    w, b = perceptron(train_vals, train_Y, max_iteration)

    prediction = []
    for sample in test_vals:
        prediction.append(predict(w, b, sample))

    print zero_one_loss(prediction, test_Y)

if __name__ == "__main__": main()