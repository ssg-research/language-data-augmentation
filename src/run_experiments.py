# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Author: Mika Juuti

import os


classifiers = ['word-lr']
#classifiers = ['char-lr','word-lr','cnn','bert']

augmentations = ['baseline','add', 'bpemb']
#augmentations = ['baseline','copy','eda','add','ppdb','gensim','bpemb','gpt2']

if 'eda' in augmentations and not os.path.exists('eda_scripts/eda.py'):
    print()
    print('Please download EDA with the bash script eda_scripts/get_eda.sh')
    print()

if 'ppdb' in augmentations and not os.path.exists('ppdb_scripts/ppdb_equivalent.txt'):
    print()
    print('Please extract ppdb_equivalent to ppdb_scripts')
    print()

def work():
    for rand in range(20200303,20200304):
    # for rand in range(20200303,20200333):
        for clf in classifiers:
            for aug in augmentations:
                print(rand)
                syscall = 'python run_experiment.py -d threat -a %s -c %s -r %d --remove_cache'%(aug,clf,rand)
                os.system(syscall)

            for aug in ['add_bpemb','add_bpemb_gpt2']:
                print(rand)
                techniques = aug.split('_')
                techniques_str = ' '.join(techniques)
                if not set(techniques).issubset(set(augmentations)):
                    print('Please run all of %s before combining augmentations.' % (techniques_str))
                    continue

                ratios = ' '.join(map(str, [1./len(techniques) for i in techniques]))
                syscall = 'python combine_augmentations.py -d threat -p 5 -t %s -r %d -n %s' % (techniques_str, rand, ratios)    
                os.system(syscall)

                syscall = 'python run_experiment.py -d threat -a %s -c %s -r %d --remove_cache'%(aug,clf,rand)
                os.system(syscall)





work()
