# Unsupervised annotators

Unsupervised annotators automatically extract terms from texts. There are 2 types of annotators: UA1 and UA2.

## Unsupervised annotator 1 (UA1)
**Unsupervised annotator 1 (UA1)** based on the IBM approach presented in the article by Fusco et al. (2022) "Unsupervised Term Extraction for Highly Technical Domains."

Term extraction is divided into two tasks: extracting multi-word expressions and single nouns.

**Multi-word expressions**
Candidates are extracted based on part-of-speech tagging using patterns consisting of nouns, adjectives and proper nouns. 
Candidates are filtered based on Topic and Specificity scores.
* **Topic Score (TS):** the cosine similarity between the multi-word expression vector and the sentence vector.
* **Specificity Score (SP)** is the average distance between multi-word expression vector and other candidates vectors.

Expressions with higher specificity scores and topics correspond to more specific terms. In the IBM paper under TS > 0.1 and SP > 0.05, a multi-word expression was reported as a term.

**Single nouns**
Morphological analysis is used to filter single nouns. A word is promoted as a term provided that its lemma corresponds to at least one heading of a verbose expression or the number of subtokens is at least four.

## Unsupervised annotator 2 (UA2)
**Unsupervised annotator 2 (UA2)** is an annotator based on Non-negative matrix factorization (NMF). 

First, the candidates are extracted and pre-cleaned of punctuation, stop-words, and digits. Next, using NMF, terms are ranked, and separated by documents and topics. After that, the top words are extracted and listed as terms.
