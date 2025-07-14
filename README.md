# TermEx

This repository provides resources for automatic term extraction from English and Kazakh texts. It includes both unsupervised and supervised annotators, as well as a manually annotated dataset.

## Unsupervised-Annotators

This folder contains two unsupervised term annotators:

* UA1 – based on morphological and semantic analysis;

* UA2 – combines semantic analysis with Non-negative Matrix Factorization (NMF).

Both annotators are designed for extracting candidate terms from English and Kazakh texts without the need for labeled data.


## Supervised_annotator

This folder includes a supervised term annotator based on the BERT-BiLSTM-CRF architecture, which integrates the BERT transformer, a bidirectional LSTM, and a Conditional Random Field layer. The model is trained on term-labeled texts using the BIO tagging scheme.

## Matcha
The Matcha dataset consists of English and Kazakh texts annotated with domain-specific terms. It covers two specialized domains: Blockchain and Materials Science.
