#!/usr/bin/env bash

# download Q&A v2.1 dataset
wget https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
wget https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz

# download conversational search dataset
wget https://msmarco.blob.core.windows.net/conversationalsearch/ann_session_train.tar.gz
wget https://msmarco.blob.core.windows.net/conversationalsearch/ann_session_dev.tar.gz

# decompress the onversational search dataset
tar -xvzf ann_session_train.tar.gz
tar -xvzf ann_session_dev.tar.gz

# remove unnecessary files
rm marco_ann_session.*.half*
rm full_marco_sessions*

# remove original tar files
rm ann_session_train.tar.gz
rm ann_session_dev.tar.gz

# download data for document title
wget https://msmarco.blob.core.windows.net/msmarcoranking/fulldocs.tsv.gz
python process.py 1
rm fulldocs.tsv.gz

# split, process data
python process.py 2

# print the statistics of the data
python process.py 3

# all done, remove all intermediate and src files
rm doctitles.tsv
rm marco_ann_session.train.all.tsv
rm marco_ann_session.dev.all.tsv
rm train_v2.1.json.gz
rm dev_v2.1.json.gz
