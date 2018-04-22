import numpy as np 
import pandas as pd 
import sys
import math
from collections import Counter

def split_label(data):
    pos = []
    neg = []
    for d in data:
        if d[-1] == 1:
            pos.append(list(d[:14]))
        else:
            neg.append(list(d[:14]))
    return pos, neg

def smooth(fv_count, num):
    fv_count['unseen'] = 0
    num_v = len(fv_count.keys())
    for key in fv_count.keys():
        fv_count[key] = math.log((fv_count[key] + 1) / float(num + num_v))
    return fv_count

def nb(pos, neg):
    len_pos = len(pos)
    len_neg = len(neg)
    total = float(len_pos + len_neg)
    pos_prob = math.log(len_pos / total)
    neg_prob = math.log(len_neg / total)

    pos_likelihood = [{}] * 14
    
    for index in range(14):
        p_feature = [ p[index] for p in pos]
        pf_count = Counter(p_feature)
        pos_likelihood[index] = smooth(pf_count, len_pos)

    neg_likelihood = [{}] * 14
    
    for index in range(14):
        n_feature = [ n[index] for n in neg]
        nf_count = Counter(n_feature)
        neg_likelihood[index] = smooth(nf_count, len_neg)

    return pos_prob, neg_prob, pos_likelihood, neg_likelihood

def predict(pos_prob, neg_prob, pos_likelihood, neg_likelihood, test_data):
    results = []
    for sample in test_data:
        pos_score = pos_prob
        neg_score = neg_prob
        for index in range(len(sample)):
            val = sample[index]
            if val in pos_likelihood[index].keys():
                pos_score += pos_likelihood[index][val]
            else:
                pos_score += pos_likelihood[index]['unseen']
            if val in neg_likelihood[index].keys():
                neg_score += neg_likelihood[index][val]
            else:
                neg_score += neg_likelihood[index]['unseen']
        
        if  pos_score >= neg_score:
            results.append(1)
        else:
            results.append(0)
    return results


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
    Fetaures: Consider the first 14 discrete attributes in yelp_cat.csv for X.
    Class label: Consider the attribute goodForGroups as the class label, i.e., Y = {0, 1}.
    """


    # sanity checks
    if len(sys.argv) < 3:
        print 'wrong number of arguments'
        return

    # import data
    try:
        raw_train = pd.read_csv(sys.argv[1], sep=',', quotechar='"', header=0, na_values='BLANK', engine='python') 
    except IOError: 
        print 'training file not found'
        return
    try:
        raw_test = pd.read_csv(sys.argv[2], sep=',', quotechar='"', header=0, na_values='BLANK', engine='python') 
    except IOError: 
        print 'testing file not found'
        return
    # get test data and training data
    test = raw_test.as_matrix()
    train = raw_train.as_matrix()
    test_X = [ list(x[:14]) for x in test]
    test_Y = [ x[-1] for x in test]

    pos, neg = split_label(train)
    a, b, c, d = nb(pos, neg)
    prediction = predict(a, b, c, d, test_X)

    print 'ZERO-ONE LOSS=%f' % (zero_one_loss(prediction, test_Y))
    
if __name__ == "__main__": main()