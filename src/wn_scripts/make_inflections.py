# Copyright (C) 2020 anonymous authors (identities withheld for blind review)
# License: see LICENSE.txt

from nltk import pos_tag, sent_tokenize, word_tokenize
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
import pickle
import os

lemmatizer = WordNetLemmatizer()

# Load pre-existing inflection dict
def load_inflections(infl_path = 'wn/inflections.pkl'):
    if os.path.isfile(infl_path):
        with open(infl_path, 'rb') as f:
            infl = pickle.load(f)
        return infl
    else:
        print("No inflection file found")

# NLTK POS to WN POS
def wn_pos(postag):
    pos_wn = []
    if postag[0]=='V':
        pos_wn = ['v']
    elif postag[0]=='N':
        pos_wn = ['n']
    elif postag[:2] =='JJ':
        pos_wn = ['a', 's']
    elif postag[:2] =='RB':
        pos_wn = ['r']
    return pos_wn

# Map lemmas and tags to word form counts
def inflection_count(text, print_every=1000):
    infl_count = defaultdict(lambda: defaultdict(lambda: Counter()))
    if type(text)==str:
        text = sent_tokenize(text)
    sent_amt = len(text)
    for i,sent in enumerate(text):
        for word, tag in pos_tag(word_tokenize(sent)):
            wn_tags = wn_pos(tag)
            for t in wn_tags:
                lemma = lemmatizer.lemmatize(word, t)
                infl_count[lemma][tag][word.lower()] += 1
        if i%print_every==0:
            print('Finding inflections:', i, '/', sent_amt)
    return infl_count

# Dict from lemmas and tags to the most common word form in an inflection count dict
def max_count(count_dict):
    max_dict = {}
    for word in count_dict:
        if word not in max_dict:
            max_dict[word] = {}
        for tag in count_dict[word]:
            max_dict[word][tag] = count_dict[word][tag].most_common()[0][0]
    return max_dict

# Make inflection dict from text corpus and save it with pickle
def save_inflection_dict(corpus, fpath):
    count_dict = inflection_count(corpus)
    count_dict_max = max_count(count_dict)
    with open(fpath, 'wb') as f:
        pickle.dump(count_dict_max, f)
