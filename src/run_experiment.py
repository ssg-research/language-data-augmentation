# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Author: Mika Juuti

import os
import pandas as pd
import numpy as np
from collections import deque
import tqdm
import joblib

# contains subroutines for training, saving/loading pipeline
import logistic_regression
from sklearn.metrics import classification_report, precision_recall_fscore_support

import data_helpers
from augmentation_helpers import augmentation_helper

# include standard modules
import argparse

# Phase 1: determine experiment parameters
class ExperimentParameters:
    dataset_name = 'threat'
    percentage = 5
    classifier_name = 'char-lr'
    aug_name = 'baseline'
    #aug_name = 'copy'
    augment_classes = [0,1]
    random_seed = 20200303

# lazy imports
def import_optional_libraries():
    if ExperimentParameters.aug_name == 'eda':
        global eda
        from eda_scripts.eda import eda
    elif ExperimentParameters.aug_name == 'add':
        global add_sentences, augmentation_helper_add
        from augmentation_helpers import add_sentences, augmentation_helper_add
    if ExperimentParameters.aug_name == 'ppdb':
        global augment_ppdb
        from ppdb_scripts.augmentation_ppdb import augment_ppdb
    elif ExperimentParameters.aug_name == 'wn':
        global augment_wn
        from wn_scripts.augmentation_wn import augment_wn
    elif ExperimentParameters.aug_name == 'gensim':
        global gensim_aug
        from embedding_scripts.augmentation_gensim import gensim_aug
    elif ExperimentParameters.aug_name == 'bpemb':
        global bpemb_aug
        from embedding_scripts.augmentation_bpemb import bpemb_aug
    elif ExperimentParameters.aug_name == 'gpt2':
        global lm_aug, LanguageModelWrapper, load_lm_corpus
        from gpt2_scripts.augmentation_lm import lm_aug, LanguageModelWrapper, load_lm_corpus

    # import classifier-specific libraries
    if ExperimentParameters.classifier_name == 'bert':
        global BertWrapper
        from fast_bert_scripts.bert_helpers import BertWrapper
    elif ExperimentParameters.classifier_name == 'cnn':
        global CNN_wrapper
        from cnn_scripts.cnn_helpers import CNN_wrapper

def run_experiment():

    import_optional_libraries()

    # Phase 1: determine experiment parameters
    assert ExperimentParameters.dataset_name in ['threat', 'identity_hate']

    ExperimentParameters.aug_dir = '../data/augmentations/%s-%03d/%s-%s'%(ExperimentParameters.dataset_name,
                                                                          ExperimentParameters.percentage,
                                                                          ExperimentParameters.aug_name,
                                                                          ''.join(list(map(str, ExperimentParameters.augment_classes))))

    ExperimentParameters.save_dir = '../results/%s-%03d/%s/%s-%s'%(ExperimentParameters.dataset_name,
                                                               ExperimentParameters.percentage,
                                                               ExperimentParameters.classifier_name,
                                                               ExperimentParameters.aug_name,
                                                               ''.join(list(map(str, ExperimentParameters.augment_classes))))

    ExperimentParameters.results_file = os.path.join(ExperimentParameters.save_dir,'%s.txt'%ExperimentParameters.random_seed)

    if ExperimentParameters.classifier_name == 'bert':
        bert_wrapper = BertWrapper(model_dir = ExperimentParameters.save_dir,
                                  data_dir = ExperimentParameters.aug_dir,
                                  results_filename = '%s'%ExperimentParameters.random_seed,
                                  random_state = ExperimentParameters.random_seed)
    else:
        bert_wrapper = None

    if ExperimentParameters.classifier_name == 'cnn':
        cnn_wrapper = CNN_wrapper(model_dir = ExperimentParameters.save_dir,
                                  data_dir = ExperimentParameters.aug_dir,
                                  results_filename = '%s'%ExperimentParameters.random_seed,
                                  random_state = ExperimentParameters.random_seed)
    else:
        cnn_wrapper = None

    # Phase 2: load datasets
    DATA_DIR = '../data/'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(os.path.join(DATA_DIR,'jigsaw-toxic-comment-classification-challenge')):
        print('Jigsaw dataset not found. ')
        print('Please download from https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge')
        print('.. and extract to the directory data-augmentation/data/jigsaw-toxic-comment-classification-challenge')
        return

    else:
        print('Found Jigsaw dataset.')


    # load train data
    data_dir = '../data/jigsaw-toxic-comment-classification-challenge'
    file = 'train.csv'
    train_csv = os.path.join(data_dir,file)

    assert os.path.exists(train_csv)

    df_train = pd.read_csv(train_csv)


    # load test data
    file = 'test.csv'
    test_csv_comments = os.path.join(data_dir,file)
    df_test_comments = pd.read_csv(test_csv_comments)

    file = 'test_labels.csv'
    test_csv_labels = os.path.join(data_dir,file)
    df_test_labels = pd.read_csv(test_csv_labels)

    # verify 'id' is unique for each row
    assert len(np.unique(df_test_labels['id'].values)) == len(df_test_labels)

    # reindex with 'id'
    df_test_labels.index = df_test_labels.id
    del df_test_labels['id']

    # verify 'id' is unique for each row
    assert len(np.unique(df_test_comments['id'].values)) == len(df_test_comments)

    # reindex with 'id'
    df_test_comments.index = df_test_comments.id
    del df_test_comments['id']

    for ind in df_test_comments.index:
        assert ind in df_test_labels.index

    for ind in df_test_labels.index:
        assert ind in df_test_comments.index

    # merge dataframes (add columns)
    df_test = pd.concat([df_test_comments, df_test_labels], axis=1)
    df_test.head()


    # drop the comments that do not have annotation
    def parse_dataset(df):
        q = '%s != -1'%ExperimentParameters.dataset_name
        df = df.query(q)
        X = df['comment_text'].\
            str.replace('\n',' ').\
            str.replace('\t',' ').\
            str.strip().\
            str.split().\
            str.join(' ')
        y = df[ExperimentParameters.dataset_name]
        return X, y


    X_train, y_train = parse_dataset(df_train)
    X_test, y_test = parse_dataset(df_test)



    # obtain bootstrapped variation of training data
    dataset_name = "%s_%s_%d" % (ExperimentParameters.dataset_name,
                                 'Percentage-%03d'%ExperimentParameters.percentage,
                                 ExperimentParameters.random_seed)


    X_train_tiny, y_train_tiny = data_helpers.get_stratified_subset(X_train, y_train, percentage=ExperimentParameters.percentage, random_state = ExperimentParameters.random_seed)
    data_helpers.save_categorical_dataset(X_train_tiny, y_train_tiny, DATA_DIR, dataset_name)


    if ExperimentParameters.aug_name == 'gpt2':
        # keep all epochs in the same folder
        load_corpus_fun = lambda : load_lm_corpus('../data/'+dataset_name+'.txt')
        print('Creating LMWrapper for dataset')
        model_save_dir = '../models/%s/%s/'%('gpt2', dataset_name)
        lm = LanguageModelWrapper(load_corpus_fun, model_save_dir = model_save_dir, random_state = ExperimentParameters.random_seed)
        if not os.path.exists(model_save_dir):
            print('Fine-tuned GPT2 path not found. Training...')
            lm.train(num_epochs = 2)
        else:
            print('Found fine-tuned GPT2 directory')

    # maps augmentation names to functions
    class Mapper:
        dict = {
            'baseline' : lambda document,num_aug: [document],
            'copy' : lambda document,num_aug: [document]*num_aug,

            # set recommended settings for small-dataset EDA
            'eda' : lambda document,num_aug: eda(document, random_state = ExperimentParameters.random_seed, alpha_sr=0.05, alpha_ri=0.05, alpha_rs=0.05, p_rd=0.05, num_aug=num_aug),

            'add': None,
            'ppdb' : lambda document,num_aug: augment_ppdb(document,num_aug, random_state = ExperimentParameters.random_seed),
            'wn' : lambda document,num_aug: augment_wn(document,num_aug, random_state = ExperimentParameters.random_seed),
            'bpemb' : lambda document,num_aug: bpemb_aug(document,num_aug, random_state = ExperimentParameters.random_seed, max_candidates = 10, incl_orig = True, min_similarity = 0., change_rate = .25),
            'gensim' : lambda document,num_aug: gensim_aug(document,num_aug, random_state = ExperimentParameters.random_seed, max_candidates = 10, incl_orig = True, min_similarity = 0., change_rate = .25),

            'gpt2' : lambda document,num_aug: lm_aug(document, num_aug, random_state = ExperimentParameters.random_seed, incl_orig=True, epoch = 2, lm_wrapper = lm),

            'add_bpemb' : None, # run combine_augmentations.py
            'add_bpemb_gpt2' : None, # run combine_augmentations.py

            # functions for training networks
            # (numpy array, numpy array) -> object
            'char-lr' : lambda x,y: logistic_regression.train_model(x, y, ngram_type='char', random_state = ExperimentParameters.random_seed),
            'word-lr' : lambda x,y: logistic_regression.train_model(x, y, ngram_type='word', random_state = ExperimentParameters.random_seed),
            'bert' : lambda x,y: bert_wrapper.train_model(x, y, num_epochs = 6),
            'cnn': lambda x, y: cnn_wrapper.train_model(x, y, num_epochs = 4)
        }

    assert ExperimentParameters.aug_name in Mapper.dict
    ExperimentParameters.aug_fun = Mapper.dict[ExperimentParameters.aug_name]

    os.makedirs(ExperimentParameters.save_dir, exist_ok=True)

    if os.path.exists(ExperimentParameters.results_file):
        print('Experiment results found.')
        with open(ExperimentParameters.results_file,'r',encoding='utf-8') as f:
            print(f.read())
        print('Exiting...')
        return
    else:
        print('Experiment results will be saved in\n\t',ExperimentParameters.results_file)


    os.makedirs(ExperimentParameters.aug_dir, exist_ok=True)
    ExperimentParameters.aug_file = os.path.join(ExperimentParameters.aug_dir,'%s.txt'%ExperimentParameters.random_seed)

    print('Augmentations will be saved in\n\t',ExperimentParameters.aug_file)

    assert ExperimentParameters.classifier_name in Mapper.dict

    ExperimentParameters.classifier_train_fun = Mapper.dict[ExperimentParameters.classifier_name]
    ExperimentParameters.classifier_file = os.path.join(ExperimentParameters.save_dir,'clf_%s.pkl'%ExperimentParameters.random_seed)

    print('Classifier will be saved in\n\t',ExperimentParameters.classifier_file)


    # run data augmentation
    if not os.path.exists(ExperimentParameters.aug_file):

        if ExperimentParameters.aug_name == 'add':
            augmentation_helper_add(input='../data/%s.txt'%dataset_name, output=ExperimentParameters.aug_file, num_aug = 20, target_classes = ExperimentParameters.augment_classes, random_state = ExperimentParameters.random_seed)
        else:
            augmentation_helper(input='../data/%s.txt'%dataset_name, output=ExperimentParameters.aug_file, aug_function = ExperimentParameters.aug_fun, num_aug = 20, target_classes = ExperimentParameters.augment_classes)


    # load augmented dataset
    X_train_aug, y_train_aug = data_helpers.load_categorical_dataset(ExperimentParameters.aug_file)
    X_train_aug = pd.Series([x.lower() for x in X_train_aug])
    print('\tDataset shapes',X_train_aug.shape, y_train_aug.shape)

    # create classifier
    if not os.path.exists(ExperimentParameters.classifier_file):
        print("Training model...")

        clf = ExperimentParameters.classifier_train_fun(X_train_aug, y_train_aug)

        print("\tSaving model to... %s"%ExperimentParameters.classifier_file)
        joblib.dump(clf, ExperimentParameters.classifier_file)


    print("Reloading Model...")
    clf = joblib.load(ExperimentParameters.classifier_file)

    print(clf.predict_proba(['if you do not stop, the wikapidea nijas will come to your house and kill you']))

    print("Predicting test data...")

    if ExperimentParameters.classifier_name == 'bert':
        y_pred = clf.predict(X_test)
    else:
        prediction_dir = '%s/%s'%(ExperimentParameters.save_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)
        prediction_file = '%s/%d-output.csv'%(prediction_dir,ExperimentParameters.random_seed)

        print('... saving prediction results to %s.'%prediction_dir)
        y_pred_prob = clf.predict_proba(X_test)
        # output from CNN is a list of lists
        y_pred_prob = np.array(y_pred_prob)
        y_pred = np.array(y_pred_prob[:,1] > 0.5, dtype = int)

        with open(prediction_file, 'w',encoding='utf-8') as f:
            for line in y_pred_prob:
                line = ', '.join(map(str,line))
                f.write('%s\n'%line)


    print(classification_report(y_test, y_pred))

    res_0 = precision_recall_fscore_support(y_test, y_pred, pos_label=0, average='binary')
    res_1 = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
    res_m = precision_recall_fscore_support(y_test, y_pred, average='macro')


    with open(ExperimentParameters.results_file,'w',encoding='utf-8') as f:
        f.write('\tPrecision\tRecall\tF-score\n')

        f.write('0:\t')
        f.write('\t'.join(map(str,res_0[:-1])))
        f.write('\n')

        f.write('1:\t')
        f.write('\t'.join(map(str,res_1[:-1])))
        f.write('\n')

        f.write('macro:\t')
        f.write('\t'.join(map(str,res_m[:-1])))
        f.write('\n')


    if args.remove_cache:
        # BERT classifier weigths take 421 MB
        if ExperimentParameters.classifier_name == 'bert':
            bert_wrapper.delete_cache()
            print('Done.')
        # word-lr takes 30 MB per file
        else:
            print('Deleting classifier file...', end='')
            os.remove(ExperimentParameters.classifier_file)
            print('Done.')

if __name__ == '__main__':

    # initiate the parser
    parser = argparse.ArgumentParser()

    # add long and short argument
    parser.add_argument("--dataset_name", "-d", help="obscene/threat", required=True)

    parser.add_argument("--percentage", "-p", help="5", type=int, default=5)

    parser.add_argument("--classifier_name", "-c", help="char-lr/char-word", required=True)

    parser.add_argument("--augmentation_name", "-a", help="baseline/copy/bpemb", required=True)

    parser.add_argument("--random_seed", "-r", help="20200303", type=int, required=True)

    parser.add_argument("--augmentation_classes", "-n", nargs='+', help="0 1 / 1", type=int, default = [1])

    parser.add_argument("--remove_cache", action='store_true')

    # read arguments from the command line
    args = parser.parse_args()

    print()
    print(args)

    # Phase 1: determine experiment parameters
    ExperimentParameters.dataset_name = args.dataset_name
    ExperimentParameters.percentage = args.percentage
    ExperimentParameters.classifier_name = args.classifier_name
    ExperimentParameters.aug_name = args.augmentation_name
    ExperimentParameters.augment_classes = args.augmentation_classes
    ExperimentParameters.random_seed = args.random_seed

    run_experiment()
    print()
