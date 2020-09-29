# 1 Introduction

This readme describes how to reproduce the experiments in this paper.

# 2 Compute hardware requirements

This code has been tested to work with a computer using
* CPU: Intel Core i9-9900K CPU @ 3.60GHz
* RAM: 32 GB
* GPU: GeForce RTX 2080 Ti (Driver Version: 435.21)

# 3 Install libraries (anaconda version)

Install Anaconda3 from https://www.anaconda.com/distribution/#linux
At the time of writing this report, the most current version is
https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

Download the latest version, e.g.
>`$ wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && bash Anaconda3-2020.02-Linux-x86_64.sh`

Create a conda environment:
>`$ conda create python==3.7.6 -n py3`

Start the virtual environment, e.g.
>`$ source activate py3`

Install pytorch and torchvision:
>`$ conda install pytorch=1.4.0 torchvision==0.5.0 -c pytorch`

Install rest of requirements
>`$ pip install -r requirements.txt`

Download spacy en_core_web_lg:
>`$ python -m spacy download en_core_web_lg`

(Optional) Verify pytorch and cuda are correctly installed:
>`$ python`

>`>> import torch`

>`>> torch.FloatTensor([1]).cuda()`

(Optional) Install apex for mixed-precision BERT
>`$ git clone https://github.com/NVIDIA/apex`

>`$ cd apex`

>`$ pip install -v --no-cache-dir ./`

# 4 Download Kaggle dataset

Download Kaggle's toxic comment classification dataset https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
and extract the directory to ./data/jigsaw-toxic-comment-classification-challenge

# 5 Supplementary code

(Optional) For running EDA, download eda.py:
>`$ cd src/eda_scripts`

>`$ bash get_eda.sh`

(Optional) For running PPDB, extract ppdb_equivalent from supplementary_data.zip:
>`$ cd ../..`

>`$ unzip supplementary_data.zip`

# 6 Run experiments in this paper

> `cd src && python run_experiments.py`
