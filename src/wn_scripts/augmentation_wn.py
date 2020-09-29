# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt

import numpy as np
import spacy

from pywsd.lesk import simple_lesk
from nltk import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

###############################################################################

# Load pre-existing inflection dict
from wn_scripts.make_inflections import load_inflections

parser = spacy.load('en_core_web_lg', disable=['entity'])
inflections = load_inflections(infl_path = 'wn_scripts/inflections.pkl')

###############################################################################

# WordNet

lemmatizer = WordNetLemmatizer()

# NLTK postag to WN postag
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

# Words to never paraphrase from WordNet (raw form)
no_synonyms = ['do', 'does', 'did', 'done', 'doing',
               'have', 'has', 'had', "'d", 'having',
               'be', 'am', "'m", 'are', "'re", "re", 'is', "'s", 'was', 'were', 'been', 'being',
               'get', 'gets', 'got', 'gotten',
               'not', "n't"]

# Document-level WordNet-augmentation
def augment_wn(document,
               num_aug=20,
               incl_orig=True,
               incl_orig_as_paraphrase=False,
               hypernyms=False,
               hyponyms=False,
               allow_lemma=False,
               change_rate=0.25,
               max_len=100,
               infl_dict=inflections,
               wsd=simple_lesk,
               sent_synonym_dict=None,
               return_synonym_dict=False,
               random_state=20200303):

    rng = np.random.RandomState(random_state)

    sents = sent_tokenize(document)
    sent_synonyms = {} if not sent_synonym_dict else sent_synonym_dict

    for i,sent in enumerate(sents):
        words = word_tokenize(sent.lower())
        words_tags = pos_tag(words)
        sent = ' '.join(words)
        sents[i] = sent

        if sent not in sent_synonyms:
            sent_synonyms[sent] = []
            for i, (word, tag) in enumerate(words_tags):
                word = word.lower()
                word_synonyms = set()
                if 0 < max_len > i and len(word)>2 and word not in no_synonyms:
                    for pos in wn_pos(tag):
                        try:
                            synset = wsd(sent, word, pos)
                        except:
                            synset = None
                        synsets = []
                        if synset:
                            synsets.append(synset)
                            if hypernyms:
                                synsets += synset.hypernyms()
                            if hyponyms:
                                synsets += synset.hyponyms()
                        for s in synsets:
                            for lemma in s.lemma_names():
                                if lemma in infl_dict and tag in infl_dict[lemma]:
                                    word_synonyms.add(infl_dict[lemma][tag])
                                elif allow_lemma:
                                    word_synonyms.add(lemma)

                sent_synonyms[sent].append([word] + sorted(list(word_synonyms-{word})))

    document_synonyms = []
    for s in sents:
        document_synonyms += sent_synonyms[s]

    synonym_ix = [i for (i,l) in enumerate(document_synonyms) if len(l)>1]

    num_possible_substitution_positions = len(synonym_ix)
    num_substitutions = num_possible_substitution_positions * change_rate
    num_substitutions = np.array(np.round(num_substitutions), dtype = int)
    num_substitutions = np.clip(num_substitutions, 1, num_possible_substitution_positions)

    doc_aug = []
    if incl_orig:
        num_aug -= 1
        doc_aug.append(document)

    for augm in range(num_aug):
        words_aug = []
        phrases_ix = rng.permutation(num_possible_substitution_positions)[:num_substitutions]
        phrases_ix = [synonym_ix[i] for i in phrases_ix]
        for i, word_list in enumerate(document_synonyms):
            if i in phrases_ix:
                word = rng.choice(word_list) if incl_orig_as_paraphrase else rng.choice(word_list[1:])
            else:
                word = word_list[0]
            words_aug.append(word)
        doc_aug.append(' '.join(words_aug))

    if return_synonym_dict:
        return doc_aug, sent_synonyms
    return doc_aug
