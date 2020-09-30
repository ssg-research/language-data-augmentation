# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see README.md
# Author: Mika Juuti

from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import os
import sys
import numpy as np
import apex
from sklearn.metrics import classification_report, precision_recall_fscore_support

import datetime
import logging

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_thresh, fbeta, roc_auc
from fast_bert.prediction import BertClassificationPredictor

def download_bert_base(pretrained_path):
    '''
    (str) -> None
    '''

    # get current directory
    current = os.getcwd()

    # make new directories
    os.makedirs(pretrained_path, exist_ok = True)

    os.chdir(pretrained_path+'/..')
    os.system("echo 'Pre-trained weights are downloaded to directory:'")

    # download the pre-trained weights
    os.system("echo 'Downloading pre-trained weights'")
    os.system("wget --no-check-certificate https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip")

    os.system("echo 'Unzipping file archive'")
    os.system("unzip uncased_L-12_H-768_A-12.zip")

    os.system("echo 'Removing orphan file'")
    os.system("rm uncased_L-12_H-768_A-12.zip")

    os.system("echo 'Returning to jupyter notebook directory'")
    os.chdir(current)

    return



class ExperimentParameters:

    MODEL_NAME = 'bert-base-uncased'
    MODEL_TYPE = 'bert'
    FINETUNED_PATH = None

    DO_TRAIN = True
    DO_EVAL = True
    DO_LOWERCASE = True

    def __init__(self, model_dir, data_dir, filename, class_name = 'threat', top_level='../../', random_state = 42):
        '''
        (str, str, str, str, str) -> None
        '''
        self.MODEL_PATH = Path(model_dir)
        self.DATA_PATH  = Path(data_dir)/filename
        print('\t\tdata_dir=',data_dir)
        self.RESULTS_FILENAME = filename+'_bert-output.csv'

        self.LABEL_COLS = [class_name]
        self.BERT_PRETRAINED_PATH = Path(top_level+'tf_bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')

        self.LABEL_PATH = Path(top_level+'data/%s-labels/'%class_name)
        self.LOG_PATH = Path(top_level+'logs/')
        #self.PRED_PATH = Path(top_level+'data/predictions/')
        self.PRED_PATH = Path(model_dir)/'predictions/'

        self.random_state = random_state
        output_dir = 'output-%d'%random_state
        self.OUTPUT_PATH = self.MODEL_PATH/output_dir
        self.DATA_PATH.mkdir(exist_ok=True)
        self.MODEL_PATH.mkdir(exist_ok=True)
        self.LOG_PATH.mkdir(exist_ok=True)
        self.OUTPUT_PATH.mkdir(exist_ok=True)
        self.LABEL_PATH.mkdir(exist_ok=True)
        self.PRED_PATH.mkdir(exist_ok=True)

        # download BERT weights
        if not self.BERT_PRETRAINED_PATH.exists():
            print('Directory',self.BERT_PRETRAINED_PATH,'not found')

            if 'uncased_L-12_H-768_A-12' in self.BERT_PRETRAINED_PATH.absolute().as_posix():
                download_bert_base(self.BERT_PRETRAINED_PATH.absolute().as_posix())
            else:
                print('Download pre-trained models from site:\nhttps://github.com/google-research/bert#pre-trained-models')

        assert os.path.exists(self.BERT_PRETRAINED_PATH)

        # setup Tokenizer
        if os.path.exists(self.BERT_PRETRAINED_PATH):
            self.tokenizer = BertTokenizer.from_pretrained(self.BERT_PRETRAINED_PATH.absolute().as_posix(), do_lower_case=self.DO_LOWERCASE)
        else:
            self.tokenizer = self.MODEL_NAME

        # write label name into directory (for fast-bert)
        with open(self.LABEL_PATH/'labels.csv','w',encoding='utf-8') as f:
            f.write(class_name)





def get_bert_args(experiment_parameters, num_epochs, mixed_precision = True, random_state = 42):
    args = Box({
        "run_text": "multilabel toxic comments with freezable layers",
        "train_size": -1,
        "val_size": -1,
        "log_path": experiment_parameters.LOG_PATH,
        "full_data_dir": experiment_parameters.DATA_PATH,
        "data_dir": experiment_parameters.DATA_PATH,
        "task_name": "toxic_classification_lib",
        "no_cuda": False,
        "bert_model": experiment_parameters.BERT_PRETRAINED_PATH,
        "output_dir": experiment_parameters.OUTPUT_PATH,
        "max_seq_length": 128, # down from 512
        "do_train": experiment_parameters.DO_TRAIN,
        "do_eval": experiment_parameters.DO_EVAL,
        "do_lower_case": experiment_parameters.DO_LOWERCASE,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": num_epochs,
        "warmup_proportion": 0.0,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": mixed_precision, # on RTX 20-series
        "fp16_opt_level": "O1", # Mixed Precision (recommended for typical use), also try
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 500,
        "logging_steps": 50,
        "eval_all_checkpoints": True,
        "overwrite_output_dir": True,
        "overwrite_cache": False,
        "seed": random_state,
        "loss_scale": 128,
        "task_name": 'intent',
        "model_name": experiment_parameters.MODEL_NAME,
        "model_type": experiment_parameters.MODEL_TYPE,
    })
    return args

def train_bert(experiment_parameters, args):

    # logging
    run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    logfile = str(experiment_parameters.LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ])

    logger = logging.getLogger()

    # cuda
    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    print()
    print('BERT training file: ',args['data_dir'],'train.csv')

    # create a fast-bert-specific data format
    torch.manual_seed(args.seed)
    databunch = BertDataBunch(args['data_dir'], experiment_parameters.LABEL_PATH,
                              experiment_parameters.tokenizer,
                              train_file='train.csv',
                              val_file=None,#'test.csv',
                              test_data='test.csv',
                              text_col="comment_text", label_col=experiment_parameters.LABEL_COLS,
                              batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'],
                              multi_gpu=args.multi_gpu, multi_label=True, model_type=args.model_type, clear_cache=False)

    metrics = []
    metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
    metrics.append({'name': 'roc_auc', 'function': roc_auc})
    metrics.append({'name': 'fbeta', 'function': fbeta})

    # create learner object
    learner = BertLearner.from_pretrained_model(databunch, args.model_name, metrics=metrics,
                                            device=device, logger=logger, output_dir=args.output_dir,
                                            finetuned_wgts_path=experiment_parameters.FINETUNED_PATH,
                                            warmup_steps=args.warmup_steps,
                                            multi_gpu=args.multi_gpu, is_fp16=args.fp16,
                                            multi_label=True, logging_steps=0)

    # train
    torch.manual_seed(args.seed)
    learner.fit(args.num_train_epochs, args.learning_rate, validate=False)

    # save
    learner.save_model()

    # free memory and exit
    del learner
    return



def predict_bert(experiment_parameters):
    # create predictor object
    predictor = BertClassificationPredictor(model_path=(experiment_parameters.MODEL_PATH/'output/model_out').absolute().as_posix(),
                                        label_path=experiment_parameters.LABEL_PATH,
                                        multi_label=True,
                                        model_type=experiment_parameters.MODEL_TYPE,
                                        do_lower_case=True)

    # predict test labels
    output = predictor.predict_batch(list(pd.read_csv(experiment_parameters.DATA_PATH/'test.csv')['comment_text'].values))

    # dump results
    pd.DataFrame(output).to_csv(experiment_parameters.PRED_PATH/experiment_parameters.RESULTS_FILENAME)

    # clean output
    preds = pd.DataFrame([{item[0]: item[1] for item in pred} for pred in output])
    print(preds.head())

    # load test data
    df_test = pd.read_csv(experiment_parameters.DATA_PATH/'test.csv')
    print(df_test.head())

    # merge dataframes
    df_pred = pd.merge(df_test, preds, how='left', left_index=True, right_index=True)
    del df_pred['comment_text']

    #df_pred = df_pred['id', 'obscene']
    df_pred['ground_truth'] = df_pred['%s_x'%LABEL_COLS[0]]
    df_pred['pred_prob'] = df_pred['%s_y'%LABEL_COLS[0]]
    del df_pred['%s_x'%LABEL_COLS[0]]
    del df_pred['%s_y'%LABEL_COLS[0]]
    print(df_pred.head())

    # write results to file
    df_pred.to_csv(experiment_parameters.PRED_PATH/experiment_parameters.RESULTS_FILENAME, index=None)
    return


def test_bert(experiment_parameters):

    df_pred = pd.read_csv(experiment_parameters.PRED_PATH/experiment_parameters.RESULTS_FILENAME)
    print(df_pred.head())

    # pick top probability
    y_pred_prob = df_pred['pred_prob'].values
    y = df_pred['ground_truth'].values
    y_pred = np.array(y_pred_prob>0.5, dtype=int)

    # print classification report and return
    print(classification_report(y, y_pred))

    res =  [list(precision_recall_fscore_support(y, y_pred,average='binary', pos_label=0))]
    res += [list(precision_recall_fscore_support(y, y_pred,average='binary', pos_label=1))]
    res += [list(precision_recall_fscore_support(y, y_pred,average='macro'))]

    res = pd.DataFrame(data=res, columns=['precision','recall','fscore','support'])

    return res



class BertWrapper(object):

    def __init__(self,
                 model_dir,
                 data_dir,
                 results_filename,
                 random_state):

        self.random_state = random_state
        # create BertExperimentParameters object
        self.experiment_parameters = lambda: ExperimentParameters(model_dir=model_dir,
                                                                  data_dir=data_dir,
                                                                  filename=results_filename,
                                                                  class_name = 'offensive',
                                                                  top_level='../',
                                                                  random_state = random_state)
        print('\t\tdata_dir=',data_dir)

        return

    def train_model(self, x, y, num_epochs = 6, mixed_precision = True):

        # create experiment parameters
        # lazy initialization
        self.experiment_parameters = self.experiment_parameters()
        self.output_path = self.experiment_parameters.OUTPUT_PATH
        self.args = get_bert_args(self.experiment_parameters,
                                  num_epochs, mixed_precision, self.experiment_parameters.random_state)

        # create pandas dataframe
        df = pd.DataFrame({'comment_text': x, 'offensive': y})
        df.to_csv(self.experiment_parameters.DATA_PATH/'train.csv')

        # train
        train_bert(self.experiment_parameters, self.args)

        # return self (wrapper that has a predict method)
        return self

    def predict_proba(self, x):
        print('\tpredicting probabilities...')
        # create pandas dataframe
        df = pd.DataFrame({'comment_text': x})
        df.to_csv(self.experiment_parameters.DATA_PATH/'test.csv')

        # create predictor object
        output_dir = 'output-%d/model_out'%self.experiment_parameters.random_state
        predictor = BertClassificationPredictor(model_path=(self.experiment_parameters.MODEL_PATH/output_dir).absolute().as_posix(),
                                            label_path=self.experiment_parameters.LABEL_PATH,
                                            multi_label=True,
                                            model_type=self.experiment_parameters.MODEL_TYPE,
                                            do_lower_case=True)

        # predict test labels
        output = predictor.predict_batch(list(pd.read_csv(self.experiment_parameters.DATA_PATH/'test.csv')['comment_text'].values))

        # dump results
        pd.DataFrame(output).to_csv(self.experiment_parameters.PRED_PATH/self.experiment_parameters.RESULTS_FILENAME)

        # clean output
        preds = pd.DataFrame([{item[0]: item[1] for item in pred} for pred in output])
        print(preds.head(5))

        y_pred_prob = preds.values.reshape(-1)
        y_pred_prob = pd.Series(y_pred_prob)
        return y_pred_prob

    def predict(self, x):
        y_pred_prob = self.predict_proba(x)
        y_pred = pd.Series(np.array(y_pred_prob>0.5, dtype=int))

        return y_pred

    # one BERT model takes ~417 MB of disk space
    # remove directory contents after predictions have been served
    def delete_cache(self):
        from shutil import rmtree
        output_dir = 'output-%d'%self.random_state
        temp_dir = self.experiment_parameters.MODEL_PATH/output_dir
        print('\tRemoving cache directory...',end='')
        rmtree(temp_dir)
        print(' Done.')
