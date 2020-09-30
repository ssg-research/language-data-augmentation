# Copyright (C) 2020 Secure Systems Group, University of Waterloo and Aalto University
# License: see LICENSE.txt
# Author: Tommi Gröndahl

import string
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nltk import word_tokenize
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class CNN(nn.Module):

    def __init__(self,
                 input_size,
                 embedding_dim=300,
                 kernel_sizes=(3, 4, 5),
                 kernel_num=10,
                 dropout=0.1,
                 class_num=2):

        super(CNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim) #embedding layer
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (size, embedding_dim)) for size in kernel_sizes]) #list of convolution layers with different kernel sizes
        self.dropout = nn.Dropout(dropout) #dropout layer
        self.fc = nn.Linear(kernel_num*len(kernel_sizes), class_num) #fully connected layer

    def forward(self, x): #Shape of x: (batch_size, sequence_length)
        emb = self.embedding(x) #Embed word sequence: (batch_size, sequence_length, embedding_dim)
        emb = emb.unsqueeze(1) #Add dimension: (batch_size, 1, sequence_length, embedding_dim)
        y_conv = [conv(emb) for conv in self.convs] #convolution for all kernel sizes: (batch_size, kernel_num, kernel_appl_num, 1) "kernel_appl_num" = how many times has the kernel fit the data based on size and stride
        y_relu = [F.relu(c).squeeze(3) for c in y_conv] #ReLu over conv-results and remove last dim: (batch_size, kernel_num, kernel_appl_num)
        y_mp = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y_relu] #max pool relu results: (batch_size, kernel_num)
        y_cat = torch.cat(y_mp, 1) #concatenate max-pooled results for all kernel sizes: (batch_size, kernel_num*len(kernel_sizes))
        y_do = self.dropout(y_cat) #dropout: (batch_size, kernel_num*len(kernel_sizes))
        o = self.fc(y_do) #apply linear layer: (batch_size, class_num)
        o = F.log_softmax(o, dim=1)
        return o


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 data,
                 labels,
                 discard_punct=False,
                 separate_punct=True,
                 lowercase=True,
                 word_to_ix=None,
                 min_occur=1,
                 vocab_size=10000,
                 print_progress=True,
                 print_description='Building dataset'):

        data_pbar = data
        if print_progress:
            data_pbar = tqdm(data)
            data_pbar.set_description(print_description)

        data_preprocessed = []

        for text in data_pbar:
            if discard_punct:
                text = ''.join([char for char in text if char not in string.punctuation])
            elif separate_punct:
                text = ' '.join(word_tokenize(text))
            if lowercase:
                text = text.lower()
            text = text.strip()
            data_preprocessed.append(text)

        self.data = data_preprocessed
        self.labels = list(labels)

        if not word_to_ix:
            vocab = [t.split() for t in self.data]
            vocab = [w for l in vocab for w in l]
            vocab = [w for (w,c) in Counter(vocab).most_common()[:vocab_size] if c>=min_occur]
            word_to_ix = {'<PAD>':0, '<UNK>':1}
            for i,w in enumerate(vocab):
                word_to_ix[w]=i+2

        self.word_to_ix = word_to_ix
        self.ix_to_word = {word_to_ix[w]:w for w in word_to_ix}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def pad_batch(batch, size, pad_ix=0):
    return [l+[pad_ix for i in range(size-len(l))] for l in batch]


def preprocess_batch(batch, word_to_ix, pad_ix=0, min_pad=5, max_pad=None):
    batch = [[word_to_ix[w] if w in word_to_ix else word_to_ix['<UNK>'] for w in t.split()][:max_pad] for t in batch]
    max_len = max([len(x) for x in batch])
    max_len = max(min_pad, max_len)
    batch = pad_batch(batch, max_len, pad_ix)
    return batch


def train_cnn(train_dataset,
              dev_dataset=None,
              pad_ix=0,
              max_pad=None,
              embedding_dim=300,
              kernel_sizes=(3, 4, 5),
              kernel_num=10,
              dropout=0.1,
              learning_rate=0.001,
              learning_rate_decay=0,
              epochs=3,
              batch_size=32,
              device='cuda',
              progress_bar=True):

    if not torch.cuda.is_available():
        device = 'cpu'

    loss_ftion = nn.NLLLoss()

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False) if dev_dataset else None

    model = CNN(input_size=len(train_dataset.word_to_ix),
                embedding_dim=embedding_dim,
                kernel_sizes=kernel_sizes,
                kernel_num=kernel_num,
                dropout=dropout,
                class_num=2)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate_decay)

    train_losses = []
    train_accuracies = []
    dev_losses = []

    for epoch in range(epochs):
        epoch += 1

        model.train() # Activate dropout
        avg_train_loss = 0
        avg_train_acc = 0

        train_pbar = train_dataloader
        if progress_bar:
            train_pbar = tqdm(train_dataloader)
            train_pbar.set_description("Training CNN: epoch " + str(epoch))

#       Training
        for batch, tgt in train_pbar:

            batch = preprocess_batch(batch, word_to_ix=train_dataset.word_to_ix, pad_ix=pad_ix, max_pad=max_pad)

            tgt = tgt.to(device)
            optimizer.zero_grad()
            clf = model(torch.tensor(batch).to(device))
            loss = loss_ftion(clf, tgt)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.data.item()
            preds = [l.tolist().index(max(l.tolist())) for l in clf]
            acc = accuracy_score(list(tgt.cpu()), preds)
            avg_train_acc += acc

#       Validation (if validation dataset is given)
        avg_dev_loss = 0

        if dev_dataloader:
            model.eval()

            dev_pbar = tqdm(dev_dataloader)
            dev_pbar.set_description('Epoch ' + str(epoch) + ' validation')

            for batch, tgt in dev_pbar:

                batch = preprocess_batch(batch, word_to_ix=train_dataset.word_to_ix, pad_ix=pad_ix, max_pad=max_pad)

                tgt = torch.tensor(tgt).to(device)
                clf = model(torch.tensor(batch).to(device))
                loss = loss_ftion(clf, tgt)
                avg_dev_loss += loss.data.item()
                preds = [l.tolist().index(max(l.tolist())) for l in clf]

        avg_train_loss = avg_train_loss / len(train_dataloader)
        avg_train_acc = avg_train_acc / len(train_dataloader)

        if dev_dataloader:
            avg_dev_loss = avg_dev_loss / len(dev_dataloader)

        if dev_losses:
            if avg_dev_loss > dev_losses[-1]:
                break
        elif train_losses and avg_train_loss > train_losses[-1]:
            break
        elif avg_train_acc == 1:
            break

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        if dev_dataloader:
            dev_losses.append(avg_dev_loss)

        print('Train loss:', avg_train_loss, "Dev loss:", avg_dev_loss)

    return model


def predict_cnn(model,
                test_dataset,
                word_to_ix,
                pad_ix=0,
                max_pad=None,
                batch_size=32,
                device='cpu',
                progress_bar=False):

    if not torch.cuda.is_available():
        device = 'cpu'

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_pbar = test_dataloader
    if progress_bar:
        test_pbar = tqdm(test_dataloader)
        test_pbar.set_description('Running test')

    model.to(device)
    model.eval()

    preds = []
    for batch, tgt in test_pbar:
        batch = preprocess_batch(batch, word_to_ix=word_to_ix, pad_ix=pad_ix, max_pad=max_pad)
        clf = model(torch.tensor(batch).to(device))
        preds += [l.tolist() for l in clf]

    return torch.exp(torch.tensor(preds)).tolist()


class CNN_wrapper(object):

    def __init__(self,
                 model_dir,
                 data_dir,
                 results_filename,
                 random_state):

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.results_filename = results_filename
        self.random_state = random_state

        self.word_to_ix = None
        self.model = None

    def train_model(self, x, y, num_epochs=2):
        train_dataset = Dataset(x, y)
        self.word_to_ix = train_dataset.word_to_ix
        self.model = train_cnn(train_dataset=train_dataset, epochs=num_epochs)
        return self

    def save_model(self, filename='model'):
        if filename.split('.')[-1] != 'pt':
            filename += '.pt'
        fpath = os.path.join(self.model_dir, filename)
        torch.save(self.model.state_dict(), fpath)

    def load_model(self, filename='model'):
        if filename.split('.')[-1] != 'pt':
            filename += '.pt'
        fpath = os.path.join(self.model_dir, filename)
        saved_state_dict = torch.load(fpath)
        input_size, embedding_dim = saved_state_dict['embedding.weight'].size()
        model = CNN(input_size=input_size, embedding_dim=embedding_dim)
        model.load_state_dict(saved_state_dict)
        self.model = model

    def predict_proba(self, x):
        default_labels = [0 for d in x]
        test_dataset = Dataset(x, default_labels, word_to_ix=self.word_to_ix)
        y_pred_prob = predict_cnn(model=self.model, test_dataset=test_dataset, word_to_ix=self.word_to_ix)
        return y_pred_prob

    def predict(self, x):
        y_pred_prob = self.predict_proba(x)
        y_pred = [l.index(max(l)) for l in y_pred_prob]
        return y_pred
