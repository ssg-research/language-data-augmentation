# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see README.md

from collections import deque
import numpy as np
import pandas as pd
import torch

from data_helpers import load_categorical_dataset

import transformers
import os

from transformers import GPT2LMHeadModel as model_class, GPT2Tokenizer as tokenizer_class


def apply(documents, fun):
    result = deque()
    for doc in documents:
        result.append(fun(doc))
    return list(result)


def filter_dataset(X, y, target_classes = [1]):
    dq = deque()
    for line, cls in zip(X,y):
        if cls in target_classes:
            line = line.\
                replace('\n',' ').\
                replace('\t',' ').\
                strip().\
                split()
            line = ' '.join(line)
            dq.append(line)
    return list(dq)


def load_lm_corpus(dataset_name, target_classes = [1]):
    X, y = load_categorical_dataset(dataset_name)
    return filter_dataset(X, y, target_classes)


class LanguageModelWrapper:

    def __init__(self,
                 load_corpus_fun,
                 model_save_dir: str,
                 random_state = 20200303):

        self.load_corpus_fun = load_corpus_fun
        self.model_save_dir = model_save_dir
        self.random_state = random_state

        # cache_dir
        self.cache_dir = './cache'

        # some default parameter values
        self.max_length = 100
        self.temperature = 1.
        self.top_k = None
        self.top_p = 0.9
        self.stop_token = '<|endoftext|>'

        corpus = self.load_corpus_fun()
        self.corpus_size = len(corpus)
        del corpus


    def train(self, num_epochs = 2):

        self.set_seed(self.random_state)

        # reload corpus, prepare all files
        # assumption: loading function does all necessary preprocessing to lines
        # corpus is a list of strings, or pandas.Series
        corpus = self.load_corpus_fun()
        self.text_dir = 'temp.txt'
        with open(self.text_dir,'w',encoding='utf-8') as f:
            for line in corpus:
                f.write('%s\n'%line)

        # run huggingface code
        #        "--fp16 O1"
        # code directly adapted from https://github.com/huggingface/transformers/blob/master/examples/run_generation.py
        syscall = ("python ./gpt2_scripts/run_language_modeling.py "
        "--model_type=gpt2 --model_name_or_path=gpt2 "
        "--do_train --overwrite_output_dir --line_by_line "
        "--per_device_train_batch_size 1 --learning_rate 2e-5 "
        " --output_dir=%s "
        " --train_data_file=%s "
        "--num_train_epochs %d "
        "--save_steps %d "
        "--seed %d")\
        %(self.model_save_dir, self.text_dir, num_epochs,
          len(corpus)*10, self.random_state)
        os.system(syscall)

        # clean up
        os.remove(self.text_dir)
        
    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed_all(seed)


    def generate(self, input: str, num_outputs: int,
                epoch_model = 2,
                device='cuda',
                length=None,
                seed = 20200505):

        with torch.no_grad():

            self.set_seed(seed)

            # ft_model_dir = os.path.join(self.model_save_dir,'checkpoint-%d'%(25*epoch_model))
            # ft_model_dir = os.path.join(self.model_save_dir,'checkpoint-%d'%(self.corpus_size*epoch_model))
            ft_model_dir = self.model_save_dir
            
            tokenizer = tokenizer_class.from_pretrained(self.model_save_dir)

            # possibly this one needs to be updated
            model = model_class.from_pretrained(ft_model_dir)
            model.to(device)

            if length is None:
                length = len(input)
            if length > self.max_length:
                length = self.max_length

            if len(input) > self.max_length:
                # find the last whole word before max_len encountered
                # to avoid splitting subwords at the start of generation
                last_space = input[:self.max_length][::-1].find(' ')
                input = input[:self.max_length-last_space]

            encoded_prompt = tokenizer.encode(input, add_special_tokens=False, return_tensors="pt")
            encoded_prompt = encoded_prompt.to(device)

            if encoded_prompt.size()[-1] == 0:
                input_ids = None
            else:
                input_ids = encoded_prompt

            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=length + len(input_ids[0]),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=1.0,
                do_sample=True,
                num_return_sequences=num_outputs,
            )

            # remove the batch dimension when returning multiple sequences
            if len(output_sequences.shape) > 2:
                output_sequences.squeeze_()

            generated_sequences = []

            for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
                generated_sequence = generated_sequence.tolist()

                # Decode text
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

                # Remove all text after the stop token
                text = text[: text.find(self.stop_token) if self.stop_token else None]

                # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
                total_sequence = (
                    # input_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                    text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
                )

                generated_sequences.append(total_sequence)

            generated_sequences = filter_dataset(generated_sequences, [1]*len(generated_sequences))

            return generated_sequences


def lm_aug(document, num_aug=20, random_state = 20200303,
           incl_orig=True, epoch = 2, lm_wrapper = None):

    assert lm_wrapper is not None

    return_vals = deque()
    if incl_orig:
        num_aug -= 1
        return_vals.append(document)

    generated = lm_wrapper.generate(document, num_aug, seed = random_state, epoch_model = epoch)
    for sent in generated:
        return_vals.append(sent)

    return list(return_vals)
