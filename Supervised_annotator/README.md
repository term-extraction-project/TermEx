# BERT-BiLSTM-CRF
BERT-BiLSTM-CRF is a supervised term annotation model designed for extracting domain-specific terms from text. It is built upon the BERT architecture and trained using labeled data in BIO format stored as TSV files.

The model consists of three main components:
1) Tokenization and Encoding: Input texts are tokenized and encoded using a pre-trained BERT model to obtain contextualized word representations.
2) Context Modeling: A bidirectional LSTM (BiLSTM) processes the BERT embeddings to capture long-range dependencies in both forward and backward directions.
3) Structured Prediction: A Conditional Random Field (CRF) layer is applied on top of the BiLSTM outputs to perform structured sequence labeling, accounting for dependencies between adjacent BIO tags.

The model was trained on annotated corpora, which were split into training, validation, and test sets in a 70:10:20 ratio.

