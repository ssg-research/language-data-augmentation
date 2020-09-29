# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt

from sklearn.model_selection import StratifiedKFold
from collections import deque
import numpy as np
import os
import pandas as pd


def get_stratified_subset(X_train, y_train, percentage=5, random_state = 20200303):
    '''
    Returns a percentage of the original data.
    Uses a deque data structure for O(1) appends.
    '''
    assert percentage > 0 and percentage <= 100
    skf_tiny = StratifiedKFold(n_splits=100, shuffle=True, random_state = random_state)
    X_return = deque()
    y_return = deque()

    # fetch non-overlapping 1 percent splits
    # take union of sets until dataset size reached
    for i, (_, tiny_train_index) in enumerate(skf_tiny.split(X_train, y_train)):

        # increase the number of samples
        X_tiny_train, y_tiny_train = X_train[tiny_train_index], y_train[tiny_train_index]
        X_return.append(X_tiny_train)
        y_return.append(y_tiny_train)

        # stop when reaching predetermined percentage
        if i+1 == percentage:
            break

    X_return = pd.concat(list(X_return))
    y_return = pd.concat(list(y_return))
    print('Input shape:',X_train.shape, '\tOutput shape:',X_return.shape)
    return X_return, y_return

# assumptions: input X and y are pandas Series
def save_categorical_dataset(X, y, directory, name):
    '''
    Saves a text dataset to separate files, in format "name.txt".
    Each row in format %d\t%s\n, where %d is the class and %s is the text.
    Typically, 0 (negative class) and 1 (positive class).
    Dataset order is retained when saved using this function.
    '''

    # ensure no newlines are in the text: each row will be a separate entry
    for x in X:
        assert '\n' not in x
        assert '\t' not in x

    fname = os.path.join(directory, '%s.txt' % (name))
    print('Writing file %s'%fname)
    with open(fname,'w',encoding='utf-8') as f:
        for x,u in zip(X,y):
            f.write('%d\t%s\n'%(int(u),x))



# return data is pandas Series
def load_categorical_dataset(fname):
    '''
    Loads a dataset (pandas.Series text entries, and pandas.Series integer entries)
    that have been saved with 'save_categorical_dataset'.
    '''
    # saves entries initially in deque for O(1) operations
    X = deque()
    y = deque()

    with open(fname,'r',encoding='utf-8') as f:
        for line in f:
            if len(line.strip().split('\t')) == 2:
                num,text = line.strip().split('\t')
            else:
                num,text = line.strip(),''
            num = np.int64(num)
            X.append(text)
            y.append(num)

    X = pd.Series(list(X))
    y = pd.Series(list(y))
    return X, y
