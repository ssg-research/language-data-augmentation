# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Author: Mika Juuti

import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def save_pipeline(pipeline, directory, name):
    joblib.dump(pipeline, os.path.join(directory, '%s_pipeline.pkl' % name))

def load_pipeline(directory, name):
    return joblib.load(os.path.join(directory, '%s_pipeline.pkl' % name))

def exists_pipeline(directory, name):
    return os.path.exists(os.path.join(directory, '%s_pipeline.pkl' % name))

def train_model(X, y, ngram_type='char', ngram_range=(1,4), vocab_size=10000, clf_C = 10, lowercase = True, random_state = 42, verbose=True):
    """
    Trains a model with sklearn logistic regression.
    Adapted from https://github.com/ewulczyn/wiki-detox/blob/master/src/modeling/get_prod_models.py
    License displayed in preamble.
    """

    assert(ngram_type in ['word', 'char'])

    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression()),
    ])

    params = {
        'vect__max_features': vocab_size,
        'vect__ngram_range': ngram_range,
        'vect__analyzer' : ngram_type,
        'vect__lowercase' : lowercase,
        'vect__strip_accents' : 'ascii',
        'tfidf__sublinear_tf' : True,
        'tfidf__norm' :'l2',
        'clf__C' : 10,
        'clf__solver' :'lbfgs',
        'clf__verbose' : 1 if verbose else 0,
        'clf__random_state' : random_state
    }

    return clf.set_params(**params).fit(X,y)
