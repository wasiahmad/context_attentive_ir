#!/usr/bin/env bash

function make_dir () {
    if [[ ! -d "$1" ]]; then
        mkdir $1
    fi
}

TMP_DIR=../tmp
DATA_DIR=../data/
FASTTEXT=../data/fasttext

for dir in $TMP_DIR $FASTTEXT;
do
	make_dir $dir;
done

echo "Downloading Fasttext word embeddings"
if [[ "$(ls -A $FASTTEXT)" ]]; then
     echo "$FASTTEXT is not empty, skipping download"
else
    # download GloVe 840B version
    curl -Lo ${FASTTEXT}/crawl-300d-2M-subword.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
    unzip ${FASTTEXT}/crawl-300d-2M-subword.zip -d ${FASTTEXT}/
    rm -f ${FASTTEXT}/crawl-300d-2M-subword.bin
fi
