#!/usr/bin/env bash

SRC_DIR=../
DATA_DIR=${SRC_DIR}/data/
EMBED_DIR=${SRC_DIR}/data/fasttext/
MODEL_DIR=${SRC_DIR}/tmp/

RGPU=$1
MODEL_NAME=$2
DATASET=msmarco


PYTHONPATH=$SRC_DIR CUDA_VISIBLE_DEVICES=$RGPU python -W ignore ${SRC_DIR}/main/multitask.py \
	--model_type $MODEL_NAME \
	--train_file train.json \
	--dev_file dev.json \
	--test_file test.json \
	--max_doc_len 200 \
	--max_query_len 20 \
	--uncase True \
	--max_examples -1 \
	--emsize 300 \
	--batch_size 32 \
	--test_batch_size 32 \
	--num_epochs 50 \
	--dropout_emb 0.2 \
	--dropout 0.2 \
	--dropout_rnn 0.2 \
	--optimizer adam \
	--learning_rate 0.001 \
	--weight_decay 0.0 \
	--early_stop 5 \
	--valid_metric bleu \
	--checkpoint True \
	--model_dir $MODEL_DIR \
	--model_name $MODEL_NAME \
	--only_test False \
	--data_workers 5 \
	--dataset_name $DATASET \
	--data_dir ${DATA_DIR}/${DATASET}/ \
	--embed_dir $EMBED_DIR \
	--embedding_file crawl-300d-2M-subword.vec
