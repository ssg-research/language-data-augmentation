# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see README.md

import numpy as np
import os
import pickle
import spacy

from collections import defaultdict
from nltk import sent_tokenize, ngrams
from itertools import chain, combinations
from string import punctuation

###############################################################################

# Required files:
# - ppdb_equivalent.txt

def ppdb_to_dict(ppdb_list, num_chars=3):
    ppdb_dict = {'comp':defaultdict(lambda: defaultdict(lambda: defaultdict(set))), 'pos':defaultdict(lambda: defaultdict(lambda: defaultdict(set)))}
    for context, phr, par in ppdb_list:
        context = context[1:-1]
        if '/' in context:
            comp = context.split('/')[1]
            ppdb_dict['comp'][phr[:num_chars]][phr][comp].add(par)
        else:
            ppdb_dict['pos'][phr[:num_chars]][phr][context].add(par)
    return ppdb_dict

def load_ppdb(path='ppdb_scripts/ppdb_equivalent.txt', num_chars=3):
    if not os.path.exists(path):
        print('-'*80)
        print(path,'not found.')
        print('Please extract ppdb_equivalent.txt from supplementary data ',end='')
        print('and add it to ./src/ppdb_scripts/')
        print('-'*80)
        raise Exception('ppdb_equivalent.txt not found')
    with open(path, 'r', encoding='utf-8') as f:
        ppdb = f.readlines()
    ppdb = [p.split('|') for p in ppdb]
    ppdb = [[x.strip() for x in l] for l in ppdb]
    ppdb = ppdb_to_dict(ppdb, num_chars=num_chars)
    return ppdb

def subsets(ls):
    return chain.from_iterable(combinations(ls, n) for n in range(len(ls)+1))

def flatten(ls):
    return [item for l in ls for item in l]

# Post-processing for correct indefinite article
vowels = 'aeio'
def fix_articles(text):
    new_text = ''
    words = text.split()
    for i in range(len(words)-1):
        word = words[i]
        next_word = words[i+1]
        if word=='a' and next_word[0] in vowels:
            new_text += 'an '
        elif word=='an' and next_word[0] not in vowels:
            new_text += 'a '
        else:
            new_text += word + ' '
    new_text += words[-1]
    return new_text

def all_ngrams(word_list):
    all_ngr = [word_list]
    for i in range(1, len(word_list)):
        all_ngr += list(ngrams(word_list, i))
    return [' '.join(ngr) for ngr in all_ngr]

def strip_punctuation(text):
    while text and text[0] in punctuation:
        text = text[1:]
    while text and text[-1] in punctuation:
        text = text[:-1]
    return text

parser = spacy.load('en_core_web_lg', disable=['entity'])
ppdb = load_ppdb()

###############################################################################

# PPDB

def all_subtrees(token, label, parser):
    subtree_list = []
    token_subtree = list(token.subtree)
    token_index = token_subtree.index(token)
    beg, end = token_subtree[:token_index+1], token_subtree[token_index+1:]
    end_str = ' '.join([t.orth_.lower() for t in end]).strip()
    # Modifiers from left to N-head; keep possible right-modifiers intact
    for i in range(len(beg)):
        beg_str = ' '.join([t.orth_.lower() for t in beg]).strip()
        if len(beg)>1:
            subtree_list.append((label, beg_str))
        if end:
            subtree_list.append((label, beg_str + ' ' + end_str))
        beg = beg[1:]
    return subtree_list

# Syntax tree information from Spacy parse, used in checking PPDB's conditions
def trees(sent, parser):
    if type(sent) == str:
        sent = parser(sent)
    sent_words = [t.orth_.lower() for t in sent]
    sent_orth = ' '.join(sent_words).strip()
    tree_list = []

    for token in sent:

        # Verb head
        if token.pos_ == 'VERB' and token.dep_ != 'aux':
            subtree_str = ' '.join([t.orth_ for t in token.subtree]).lower()
            tree_list.append(('VB', token.orth_))

            # S/SBAR: Full sentences
            if token.dep_ == 'ROOT':
                tree_list.append(('S', subtree_str))
            elif token.dep_ == 'ccomp':
                tree_list.append(('SBAR', subtree_str))

            # VP: Sentence without subject
            subjects = [t for t in sent if t.head == token and t.dep_ == 'nsubj']
            vp_subtree = list(token.subtree)
            subject_parts = set()
            for subj in subjects:
                subject_parts = subject_parts | set(subj.subtree)
            vp_subtree = [t for t in vp_subtree if t not in subject_parts]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))

            # VP: Verb + object, no modifiers except aux + negation
            aux_neg = [t for t in sent if t.head == token and t.dep_ in ['aux', 'neg']]
            objects = [t for t in sent if t.head == token and t.dep_ in ['dobj', 'dative']]
            object_parts = set()
            for obj in objects:
                object_parts = object_parts | set(obj.subtree)
            vp_subtree = [t for t in vp_subtree if t==token or t in object_parts | set(aux_neg)]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))

            # VP: Verb + object, no modifiers
            vp_subtree = [t for t in vp_subtree if t==token or t in object_parts]
            tree_list.append(('VP', ' '.join([t.orth_.lower() for t in vp_subtree]).strip()))

            # VP: modifiers + verb, no subject or object
            vp_subtree = [t for t in token.subtree if t not in subject_parts | object_parts]
            token_index = vp_subtree.index(token)
            beg, end = vp_subtree[:token_index+1], vp_subtree[token_index:][::-1]
            for i in range(len(beg)):
                tree_list.append(('VP', ' '.join([t.orth_.lower() for t in beg]).strip()))
                beg = beg[1:]
            for i in range(len(end)):
                tree_list.append(('VP', ' '.join([t.orth_.lower() for t in end][::-1]).strip()))
                end = end[1:]

        else:
            if token.pos_=='NOUN':
                tree_list.append(('NN', token.orth_))
                tree_list += all_subtrees(token, 'NP', parser=parser)

            elif token.pos_=='ADJ':
                tree_list.append(('JJ', token.orth_))
                tree_list += all_subtrees(token, 'ADJP', parser=parser)

            elif token.pos_=='ADJ':
                tree_list.append(('JJ', token.orth_))
                tree_list += all_subtrees(token, 'ADJP', parser=parser)

            elif token.pos_=='ADV':
                tree_list.append(('RB', token.orth_))
                tree_list += all_subtrees(token, 'ADVP', parser=parser)

            elif token.pos_=='ADP':
                subtree_str = ' '.join([t.orth_ for t in token.subtree]).lower()
                tree_list.append(('PP', subtree_str))

    tree_list = [pair for pair in tree_list if pair[1] in sent_orth]
    tree_list +=[(label, strip_punctuation(s)) for (label, s) in tree_list]


    return tree_list

# Map phrases in a sentence to paraphrases from PPDB
def ppdb_matches(ppdb_dict,
                 sent,
                 parser,
                 sent_parsed=None,
                 sent_words=None,
                 max_len=100,
                 single_words=True,
                 filter_ungrammatical=False,
                 ppdb_dict_key_len=3):

#    pars = defaultdict(list)
    pars = defaultdict(set)
    if not sent_parsed:
        sent_parsed = parser(sent.lower())
    if not sent_words:
        sent_words = [t.orth_ for t in sent_parsed]
    sent = ' '.join(sent_words)
    sent_trees = trees(sent_parsed[:max_len], parser=parser)

    for ngr in all_ngrams(sent_words[:max_len]):
        start = sent.index(ngr)
        end = start + len(ngr)
        complement = sent[end:].strip()

        if ngr in ppdb_dict['comp'][ngr[:ppdb_dict_key_len]]:
            for required_label in ppdb_dict['comp'][ngr[:ppdb_dict_key_len]][ngr]:
                for label, phrase in sent_trees:
                    if phrase in complement and phrase[:10]==complement[:10] \
                    and (label == required_label or (label[0] in ['N', 'V'] and label[0]==required_label[0])):
                        pars[ngr] = pars[ngr] | ppdb_dict['comp'][ngr[:ppdb_dict_key_len]][ngr][required_label]

    if single_words:
        for label, phrase in sent_trees:
            char_key = phrase[:ppdb_dict_key_len]
            if phrase in ppdb_dict['pos'][char_key]:
                if label in ppdb_dict['pos'][char_key][phrase]:
#                    pars[phrase] += ppdb_dict['pos'][char_key][phrase][label]
                    pars[phrase] = pars[phrase] | ppdb_dict['pos'][char_key][phrase][label]

    pars = {phr:sorted(list(pars[phr])) for phr in pars}

    # Filter out ungrammatical paraphrases
    # sort to ensure consistency when iterating dict keys
    if filter_ungrammatical:
        for phr in sorted(list(pars.keys())):
            pars_filter = sorted(list(pars[phr]))
            phr_parsed = parser(phr)
            phr_words = []
            phr_tags = set()
            phr_prons = set()
            for t in phr_parsed:
                phr_words.append(t.orth_)
                phr_tags.add(t.tag_)
                if t.pos_=='PRON' or t.tag_=='PRP$':
                    phr_prons.add(t.orth_)

            for par in sorted(list(pars[phr])):
                if par in pars_filter:
                    par_parsed = parser(par)
                    par_words = []
                    par_tags = set()
                    par_prons = set()
                    for t in par_parsed:
                        par_words.append(t.orth_)
                        par_tags.add(t.tag_)
                        if t.pos_ == 'PRON' or t.tag_=='PRP$':
                            par_prons.add(t.orth_)

                    # No pronoun changes
                    if phr_prons != par_prons:
                        pars_filter.remove(par)
                    # First and last tags must be the same
                    for i in [0, -1]:
                        if par in pars_filter and par_parsed[i].tag_ != phr_parsed[i].tag_:
                            pars_filter.remove(par)
                    # Tense/number inflection
                    for tag in ['VB', 'VBP', 'VBD', 'VBG', 'VBN', 'VBZ', 'NN', 'NNP', 'NNPS', 'NNS']:
                        if par in pars_filter and tag in (phr_tags - par_tags) | (par_tags - phr_tags):
                            if not (tag in ['VB', 'VBP'] and {'VB', 'VBP'} & par_tags):
                                pars_filter.remove(par)
                    # Copula inflection with "I"
                    for i, word in enumerate(par_words):
                        if word in ['am', 'are'] and word not in phr_words and (i==0 or par_words[i-1]!='i'):
                            prev = sent[:sent.index(phr)].split()
                            if par in pars_filter and ((word=='am' and 'i' not in prev) or (word=='are' and 'i' in prev)):
                                pars_filter.remove(par)
            pars[phr] = pars_filter
    pars = {p:sorted(pars[p]) for p in pars if pars[p]}
    return pars

# Document-level PPDB-augmentation
def augment_ppdb(document,
                 num_aug=20,
                 incl_orig=True,
                 incl_orig_as_paraphrase=False,
                 change_rate=0.25,
                 max_len=100,
                 ppdb_dict=ppdb,
                 parser=parser,
                 pars_dict=None,
                 return_pars_dict=False,
                 random_state=20200303):

    rng = np.random.RandomState(random_state) # numpy-random

    sents = sent_tokenize(document)
    sent_pars = {} if not pars_dict else pars_dict

    for i,sent in enumerate(sents):
        sent_parsed = parser(sent)
        sent_words = [t.orth_.lower() for t in sent_parsed]
        sent = ' '.join(sent_words).strip()
        sents[i] = sent

        if sent not in sent_pars:
            pars = ppdb_matches(ppdb_dict=ppdb_dict, sent=sent, parser=parser, sent_parsed=sent_parsed, sent_words=sent_words, max_len=max_len, single_words=True)
            sent_pars[sent] = pars

    sent_pars_list = []
    for i,s in enumerate(sents):
        for p in sent_pars[s]:
            par_list = sent_pars[s][p]
            if incl_orig_as_paraphrase:
                par_list.append(p)
            sent_pars_list.append((i, p, par_list))

    num_possible_substitution_positions = len(sent_pars_list)
    num_substitutions = num_possible_substitution_positions * change_rate
    num_substitutions = np.array(np.round(num_substitutions), dtype = int)
    num_substitutions = np.clip(num_substitutions, 1, num_possible_substitution_positions)

    doc_aug = []
    if incl_orig:
        num_aug -= 1
        doc_aug.append(document)

    for augm in range(num_aug):
        phrases_ix = rng.permutation(num_possible_substitution_positions)[:num_substitutions]
        paraphrases = [sent_pars_list[i] for i in phrases_ix]
        sents_aug = list(sents)
        for sent_ix, phrase, paraphrase_list in paraphrases:
            sent_aug = sents_aug[sent_ix].replace(phrase, rng.choice(paraphrase_list))
            sents_aug[sent_ix] = sent_aug
        doc_aug.append(' '.join(sents_aug))

    if return_pars_dict:
        return doc_aug, sent_pars
    return doc_aug
