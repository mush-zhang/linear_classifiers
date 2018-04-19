"""
split the dataset into train and test 

"""
import pandas as pd 
import sys
import math

"""
argv[0]: program name
argv[1]: dataset file
argv[2]: percentage of training
"""

# sanity checks
if len(sys.argv) < 3:
    print 'wrong number of arguments'
    sys.exit

try:
    perc = float(sys.argv[2]) / 100
except ValueError:
    print 'percentage is not an number'
    sys.exit
if perc < 0:
    print 'percentage should be non negative'
    sys.exit

# import data
try:
    raw = pd.read_csv(sys.argv[1], sep=',', quotechar='"', header=0, engine='python') 
except IOError: 
    print 'training file not found'
    sys.exit

# shuffle data
sh_raw = raw.sample(frac=1).reset_index(drop=True)

# length of train
train_len = int(round(len(raw) * perc))

# write to seperate files
sh_raw[:train_len].to_csv('train.csv', sep=',', index=False)
sh_raw[train_len:].to_csv('test.csv', sep=',', index=False)
