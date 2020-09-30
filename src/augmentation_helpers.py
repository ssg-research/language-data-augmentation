# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Authors: Mika Juuti and Tommi Gröndahl

from nltk import sent_tokenize
from collections import deque

import numpy as np
import tqdm


def augmentation_helper(input, output, aug_function, num_aug=20, target_classes = [0,1]):
    with open(input,'r',encoding='utf-8') as f:
        for total,_ in enumerate(f):
            pass

    # write files temporarily into a deque
    dq = deque()
    with open(input,'r',encoding='utf-8') as f:
        for line in tqdm.tqdm(f, total=total):
            id,sent = line.strip().split('\t')
            if int(id) in target_classes:
                aug_sents = aug_function(sent, num_aug=num_aug)
            else:
                aug_sents = [sent]
            for sent in aug_sents:
                dq.append('%s\t%s\n'%(id,sent))

    with open(output,'w',encoding='utf-8') as fo:
        fo.write(''.join(list(dq)))


# Sentence addition
def add_sentences(document,
                  add_sentence_corpus,
                  num_aug=20,
                  max_replacements=1,
                  incl_orig=True,
                  rng=None):

    if not rng:
        rng = np.random.RandomState(20200303)

    sents = sent_tokenize(document)

    num_add = rng.randint(1, max_replacements+1)
    add_range = len(add_sentence_corpus)

    doc_aug = []
    if incl_orig:
        doc_aug = [document]
        num_aug -= 1

    for aug in range(num_aug):

        sents_add = list(sents)

        for i in range(num_add):
            add_position = rng.randint(len(sents_add)+1)
            add_sent_ix = rng.randint(add_range)
            add_sent = add_sentence_corpus[add_sent_ix]
#            add_sent = rng.choice(add_sentence_corpus)
            sents_add.insert(add_position, add_sent)

        doc_aug.append(' '.join(sents_add))

    return doc_aug

# Sentence addition for whole corpus (not resetting rng for every document)
def augmentation_helper_add(input,
                             output,
                             num_aug=20,
                             add_from_classes=[0],
                             target_classes=[0,1],
                             random_state=20200303):

    rng = np.random.RandomState(random_state)

    with open(input,'r',encoding='utf-8') as f:
        for total,_ in enumerate(f):
            pass

    dq = deque()

    with open(input,'r',encoding='utf-8') as f:

        # Create corpus to add sentences from
        add_sentence_corpus = []
        for line in f:
            label, doc = line.strip().split('\t')
            if int(label) in add_from_classes and doc.strip():
                add_sentence_corpus += sent_tokenize(doc)

    with open(input,'r',encoding='utf-8') as f:

        for line in  tqdm.tqdm(f, total=total):
            label, doc = line.strip().split('\t')
            if int(label) in target_classes:
                aug_sents = add_sentences(document=doc, add_sentence_corpus=add_sentence_corpus, num_aug=num_aug, rng=rng)
            else:
                aug_sents = [doc]

            for sent in aug_sents:
                dq.append('%s\t%s\n'%(label,sent))
                #fo.write('%s\t%s\n'%(id,sent))

    with open(output,'w',encoding='utf-8') as fo:
        fo.write(''.join(list(dq)))
