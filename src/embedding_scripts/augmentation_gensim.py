# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see README.md
# Authors: Mika Juuti and Tommi Gröndahl

import numpy as np
import time
from nltk import word_tokenize
from collections import deque

import gensim.downloader as api

# Static storage
# Load pre-trained GloVe embeddings trained on Twitter
class Storage:
    #print('Loading GloVe embeddings...,', end=' ')
    twitter25 = api.load("glove-twitter-25")
    #print('Done!')

def gensim_aug(document, num_aug=20, gensim_model=Storage.twitter25,
               rng=None, random_state = 20200303,
               max_candidates = 10, incl_orig=True, incl_orig_as_paraphrase=False,
               change_rate = 0.25, min_similarity = 0.):

    change_rate = np.clip(change_rate, 0., 1.)

    if not rng:
        rng = np.random.RandomState(seed = random_state)

    doc_words = word_tokenize(document.lower())
    possible_substitution_positions = [i for (i,w) in enumerate(doc_words) if w in gensim_model]

    num_possible_substitution_positions = len(possible_substitution_positions)
    num_substitutions = num_possible_substitution_positions * change_rate
    num_substitutions = np.array(np.round(num_substitutions), dtype = int)

    # at least one substitution
    num_substitutions = np.clip(num_substitutions, 1, num_possible_substitution_positions)

    if num_substitutions * max_candidates < num_aug:
        max_candidates = min(10, np.round(num_aug / num_substitutions))

    max_candidates = int(max_candidates)

    return_vals = deque()
    if incl_orig:
        num_aug -= 1
        return_vals.append(document)

    # pre-compute most similar words
    most_similar_dict = {}
    for w in doc_words:
        most_similar = [w] if incl_orig_as_paraphrase else []
        if w in gensim_model:
            for sim_w, score in gensim_model.most_similar(w, topn=max_candidates):
                if score > min_similarity:
                    most_similar.append(sim_w)
        if most_similar == []:
            most_similar = [w]
        most_similar_dict[w] = most_similar

#    old words to be replaced
    substitute_positions = deque()
    for _ in range(num_aug):
        substitute_positions.append(sorted(rng.permutation(num_possible_substitution_positions)[:num_substitutions]))
    substitute_positions = sorted(substitute_positions)
#
#    # new words
    substituted_similar_subwords = deque()
    for _ in range(num_aug):
        temp_list = []
        for _ in range(num_substitutions):
            temp_list.append(rng.randint(max_candidates))
        substituted_similar_subwords.append(temp_list)

    for batch_positions, batch_substitutions in zip(substitute_positions, substituted_similar_subwords):
        temp_list = doc_words.copy()
        for position_idx, substitution_idx in zip(batch_positions, batch_substitutions):
            sw = doc_words[position_idx]
            sw_substitution = most_similar_dict[sw][substitution_idx % len(most_similar_dict[sw])]
            temp_list[position_idx] = sw_substitution
        return_vals.append(' '.join(temp_list))

    return return_vals
