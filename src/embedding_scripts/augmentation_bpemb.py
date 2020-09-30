# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see README.md
# Author: Mika Juuti

from bpemb import BPEmb
from collections import deque
import numpy as np
import tqdm

# Static storage
class Storage:
    bpemb = BPEmb(lang="en", dim=50, vs=10000)


def apply(documents, fun):
    result = deque()
    for doc in documents:
        result.append(fun(doc))
    return list(result)

def bpemb_aug(document, num_aug=20, rng=None, random_state = 20200303,
                max_candidates = 10, incl_orig=True, change_rate = 0.25,
                min_similarity = 0.):


    change_rate = np.clip(change_rate, 0., 1.)

    if not rng:
        rng = np.random.RandomState(seed = random_state)

    subwords = Storage.bpemb.encode(document)

    num_possible_substitution_positions = len(subwords)
    num_substitutions = num_possible_substitution_positions * change_rate
    num_substitutions = np.array(np.round(num_substitutions), dtype = int)

    if num_substitutions * max_candidates < num_aug:
        max_candidates = min(10, np.round(num_aug / num_substitutions))

    max_candidates = int(max_candidates)

    # at least one substitution
    num_substitutions = np.clip(num_substitutions, 1, num_possible_substitution_positions)

    return_vals = deque()
    if incl_orig:
        num_aug -= 1
        return_vals.append(document)

    # pre-compute most similar subwords
    most_similar_dict = {}
    for sw in subwords:
        approved_similar = [sw]
        try:
            most_similar, scores = zip(*Storage.bpemb.most_similar(sw))
            approved_similar = []
            # print(sw)
            for a,b in zip(most_similar, scores):
                if b > min_similarity:
                    approved_similar.append(a)
            most_similar = approved_similar[:max_candidates]
        except:
            most_similar = [sw]
        if most_similar == []:
            most_similar = [sw]
        most_similar_dict[sw] = most_similar

    # old subwords that are replaced
    substitute_positions = deque()
    for _ in range(num_aug):
        substitute_positions.append(sorted(rng.permutation(num_possible_substitution_positions)[:num_substitutions]))
    substitute_positions = sorted(substitute_positions)

    # new subwords
    substituted_similar_subwords = deque()
    for _ in range(num_aug):
        temp_list = []
        for _ in range(num_substitutions):
            temp_list.append(rng.randint(max_candidates))
        substituted_similar_subwords.append(temp_list)

    for batch_positions, batch_substitutions in zip(substitute_positions, substituted_similar_subwords):
        temp_list = subwords.copy()
        for position_idx, substitution_idx in zip(batch_positions, batch_substitutions):
            sw = subwords[position_idx]
            sw_substitution = most_similar_dict[sw][substitution_idx % len(most_similar_dict[sw])]
            temp_list[position_idx] = sw_substitution
        return_vals.append(Storage.bpemb.decode(temp_list))

    return list(return_vals)



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



def bpemb_augment(input, output, num_aug=20, target_classes = [0,1]):
    with open(input,'r',encoding='utf-8') as f:
        for total,_ in enumerate(f):
            pass


    with open(input,'r',encoding='utf-8') as f, open(output,'w',encoding='utf-8') as fo:
        for line in tqdm.tqdm(f, total=total):
            id,sent = line.strip().split('\t')
            if int(id) in target_classes:
                aug_sents = bpemb_aug(sent, num_aug=num_aug)
            else:
                aug_sents = [sent]
            for sent in aug_sents:
                dq.append('%s\t%s\n'%(id,sent))
                fo.write('%s\t%s\n'%(id,sent))

    fo.write(list(dq))
