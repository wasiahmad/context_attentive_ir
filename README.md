## Context-aware Neural Information Retrieval

### Introduction

PyTorch code for our ICLR 2018 and SIGIR 2019 papers.
  - [ICLR 2018] [Multi-task Learning for Document Ranking and Query Suggestion](https://openreview.net/pdf?id=SJ1nzBeA-)
  - [SIGIR 2019] [Context Attentive Document Ranking and Query Suggestion](https://arxiv.org/abs/1906.02329)

The codebase contains source-code of 8 document ranking models, 3 query suggestions models and 3 multi-task context-aware ranking and suggestion models. 

##### Document Ranking Models
- ESM: Embedding Space Model [See the details in our ICLR'18 paper]
- DSSM: [Learning Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)
- CDSSM: [Learning Semantic Representations Using Convolutional Neural Networks for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/www2014_cdssm_p07.pdf)
- DRMM: [A Deep Relevance Matching Model for Ad-hoc Retrieval](https://arxiv.org/abs/1711.08611)
- ARCI/ARCII: [Convolutional Neural Network Architectures for Matching Natural Language Sentences](https://arxiv.org/pdf/1503.03244.pdf)
- DUET: [Learning to Match using Local and Distributed Representations of Text for Web Search](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf)
- MATCH TENSOR: [Match-Tensor: a Deep Relevance Model for Search](https://arxiv.org/abs/1701.07795)

##### Query Suggestion Models

- HREDQS: [A Hierarchical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion](https://arxiv.org/abs/1507.02221)
- Seq2seq with Attention: [Effective Approaches to Attention-based Neural Machine Translation](https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- ACG: [Learning to Attend, Copy, and Generate for Session-Based Quer Suggestion](https://arxiv.org/abs/1708.03418)

Please note, we have a simplified implementation of ACG.

##### Multi-task Learning Models

- MNSRF/M-MATCH-TENSOR: [Multi-task Learning for Document Ranking and Query Suggestion](https://openreview.net/pdf?id=SJ1nzBeA-)
- CARS: [Context Attentive Document Ranking and Query Suggestion](https://arxiv.org/abs/1906.02329)

### Requirements

* python 3.6
* pytorch >= 0.4 (tested on pytorch 0.4.1)
* [spaCy](https://spacy.io/usage)
* [tqdm](https://pypi.org/project/tqdm/)
* [prettytable](https://pypi.org/project/PrettyTable/)


### Training/Testing Models

```
$ cd  scripts
$ bash SCRIPT_NAME GPU_ID MODEL_NAME
```

- To train/test document ranking models, use `ranker.sh` in place of `SCRIPT_NAME`
- To train/test query suggestion models, use `recommender.sh` in place of `SCRIPT_NAME`
- To train/test multitask models, use `multitask.sh` in place of `SCRIPT_NAME`

Here is a list of models which you can use in place of `MODEL_NAME`.

- Document Ranking Models: `esm, dssm, cdssm, drmm, arci, arcii, duet, match_tensor`
- Query Suggestion Models: `seq2seq, hredqs, acg`
- Multitask Models: `mnsrf, m_match_tensor, cars`

For example, if you want to run our CARS model, run the following command.

```
bash multitask.sh GPU_ID cars
```

##### Running experiments on CPU/GPU/Multi-GPU

- If `GPU_ID` is set to -1, CPU will be used.
- If `GPU_ID` is set to one specific number, only one GPU will be used.
- If `GPU_ID` is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.

### An Artificial Dataset

We are unable to make our experimental dataset publicly available. However, we are sharing scripts to create an artificial dataset from [MSMARCO Q&A v2.1](https://github.com/microsoft/MSMARCO-Question-Answering#qa) and [MSMARCO Conversational Search](https://github.com/microsoft/MSMARCO-Conversational-Search#corpus-generation) datasets. Please run the [script](https://github.com/wasiahmad/context_attentive_ir/blob/master/data/msmarco/get_data.sh) by going into the `/data/msmarco/` directory. Once the data is generated, you should be able to see a table showing the following statistics.

| Attribute           |   Train |  Dev   |   Test |
| :--- | ---: | ---: | ---: |
| Sessions            |  223876 | 24832  |  27673 |
| Queries             | 1530546 | 169413 | 189095 |
| Avg Session Len     |    6.84 |  6.82  |   6.83 |
| Avg Query Len       |    3.84 |  3.85  |   3.84 |
| Max Query Len       |      40 |   32   |     32 |
| Avg Doc Len         |   63.41 | 63.43  |  63.48 |
| Max Doc Len         |     290 |  290   |    290 |
| Avg Click Per Query |    1.05 |  1.05  |   1.05 |
| Max Click Per Query |       6 |   6    |      6 |

### Results on the Artificial Dataset

Coming soon!

### Acknowledgement

I borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py). I would like to expresse my gratitdue for authors of these repositeries.


### Citation

If you find the resources in this repo useful, please cite our works.

```
@inproceedings{Ahmad:2019:CAD:3331184.3331246,
 author = {Ahmad, Wasi Uddin and Chang, Kai-Wei and Wang, Hongning},
 title = {Context Attentive Document Ranking and Query Suggestion},
 booktitle = {Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
 year = {2019},
 pages = {385--394}
} 
```

```
@inproceedings{uddin2018multitask,
 title={Multi-Task Learning for Document Ranking and Query Suggestion},
 author={Wasi Uddin Ahmad and Kai-Wei Chang and Hongning Wang},
 booktitle={International Conference on Learning Representations},
 year={2018}
}
```
