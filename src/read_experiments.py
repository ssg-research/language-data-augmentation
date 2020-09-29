# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt

import os
import pandas as pd

# Phase 1: determine experiment parameters
class ExperimentParameters:
    dataset_name = 'threat'
    percentage = 5
    classifier_name = 'char-lr'
    aug_name = 'baseline'
    aug_name = 'copy'
    random_seed = 20200303

def parse_file(filename):
    with open(filename,'r',encoding='utf-8') as file:
        file.readline()
        _,a,b,_ = file.readline().rstrip().split('\t')
        _,c,d,_ = file.readline().rstrip().split('\t')
        _,_,_,e = file.readline().rstrip().split('\t')
        res = [c,d,a,b,e]
        return list(map(float,res))

def read_results(results_dir):
    columns=['off precision', 'off recall', 'non-off precision', 'non-off recall', 'macro-averaged F1']
    df = pd.DataFrame(columns=columns)
    for r,d,fs in os.walk(results_dir):
        for filename in sorted(fs):
            if filename[-3:] == 'txt':
                row = parse_file(os.path.join(r,filename))
                df2 = pd.DataFrame(data=[row],columns=columns, index=[int(filename[:-4])])
                df  = df.append(df2)
    return df


results_dir = '../results/%s-%03d/%s/%s-1'%(ExperimentParameters.dataset_name,
                                             ExperimentParameters.percentage,
                                             ExperimentParameters.classifier_name,
                                             ExperimentParameters.aug_name)
print(results_dir)

df = read_results(results_dir)

m = df.mean()
s = df.std()

for ind,m_,s_ in zip(m.index, m, s):
    print('%s: %.2f +/- %.2f'%(ind, m_, s_))
