# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Author: Mika Juuti

import pandas as pd
import os
import numpy as np
import argparse

def combine_augmentations(techniques = ['add','bpemb','gpt2'],
                          probability = [0.33,0.33,0.34],
                          dataset_name = 'threat',
                          random_seed = 20200303,
                          percentage = 5):


    new_augmentation_name = '_'.join(techniques)
    print(new_augmentation_name, random_seed)
    
    df = pd.DataFrame()
    prev_aug_files = []
    
    for aug_name in techniques:
        
        aug_dir = '../data/augmentations/%s-%03d/%s-1'%(dataset_name,
                                                        percentage,
                                                        aug_name)
        
        prev_aug_files += []
        
        with open('%s/%s.txt'%(aug_dir,random_seed), 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = pd.Series(lines)
            df2 = pd.DataFrame(data=lines, index=np.arange(len(lines)), columns=[aug_name])
            if len(df):
                assert len(df) == len(df2)
                df = df.join(df2)
            else:
                df = df2
                
    new_aug_dir = '../data/augmentations/%s-%03d/%s-1'%(dataset_name,
                                                        percentage,
                                                        new_augmentation_name)
    
    os.makedirs(new_aug_dir, exist_ok = True)
    np.random.seed(random_seed)
    
    with open('%s/%s.txt'%(new_aug_dir,random_seed), 'w', encoding='utf8') as f:
        for row in df.values:
            c = np.random.choice(np.arange(len(techniques)))
            f.write('%s'%(row[c]))

            
if __name__ == '__main__':

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    
    parser.add_argument("--dataset_name", "-d", help="obscene/threat", required=True)

    parser.add_argument("--percentage", "-p", help="5", type=int, default=5)

    parser.add_argument("--techniques", "-t", nargs='+', help="add bpemb gpt2", type=str, required=True)

    parser.add_argument("--random_seed", "-r", help="20200303", type=int, required=True)

    parser.add_argument("--ratios", "-n", nargs='+', help="0.33 0.33 0.34", type=float, required=True)

    # read arguments from the command line
    args = parser.parse_args()

    print(args)

    combine_augmentations(techniques = args.techniques,
                          probability = args.ratios,
                          dataset_name = args.dataset_name,
                          random_seed = args.random_seed,
                          percentage = 5)
