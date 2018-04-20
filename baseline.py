import numpy as np 
import pandas as pd 
import sys
import math


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

def predict(weight, bias, sample):
    score = np.sum(np.dot(weight, sample)) + bias #* len(sample)
    if score >= 0:
        return 1
    else:
        return 0

def main():

    """
    argv[0]: program name
    argv[1]: training file
    argv[2]: test file
    Fetaures: Consider the first 14 discrete attributes in yelp_cat.csv for X.
    Class label: Consider the attribute goodForGroups as the class label, i.e., Y = {0, 1}.
    """


    # sanity checks
    if len(sys.argv) < 3:
        print 'wrong number of arguments'
        return

    # import data
    try:
        raw_train_Y = pd.read_csv(sys.argv[1], sep=',', quotechar='"', header=0, usecols=['goodForGroups'], engine='python')
    except IOError: 
        print 'training file not found'
        return
    try:
        raw_test_Y = pd.read_csv(sys.argv[2], sep=',', quotechar='"', header=0, usecols=['goodForGroups'], engine='python') 
    except IOError: 
        print 'testing file not found'
        return
    # get test data and training data
    train = [ x[0] for x in raw_train_Y.as_matrix()]
    test = [ x[0] for x in raw_test_Y.as_matrix()]

    pos = 0
    neg = 0
    for d in train:
        if d == 1:
            pos += 1
        else:
            neg += 1

    if pos >= neg:
        prediction = [1] * len(test)
    else: 
        prediction = [0] * len(test)        

    print zero_one_loss(prediction, test)
    

if __name__ == "__main__": main()