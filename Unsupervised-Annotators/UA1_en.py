"""
Installing necessary modules and run necessary scripts
"""

!! pip install torch
!! pip install spacy
!! python -m spacy download en_core_web_sm
!! pip install sentence-transformers
! git clone https://github.com/AylaRT/ACTER.git

from torch import Tensor
import os
import pandas as pd

import torch
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')                                            # Load the language model
infixes = [ pattern for pattern in nlp.Defaults.infixes if pattern != r"-" ]  # Remove hyphen from the default infix patterns
infix_regex = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
# model = SentenceTransformer('distilbert-base-nli-mean-tokens', device=device)
# model = SentenceTransformer('deberta-v3-base', device=device)

from sentence_transformers import SentenceTransformer, util
from typing import List
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")                # Initialize the BERT tokenizer

from collections import namedtuple
CandidateMWE = namedtuple('CandidateMWE',['text','head', 'sentence','self_encode', 'sent_encode'])
CandidateW = namedtuple('CandidateW',['text','lemma', 'self_encode' ])

"""
Reading files with texts and true terms from the ACTER corpus and converting them to txt format
"""

# Import dataset from ACTER

folder_test = "/content/tests/"    # Path where ACTER files will be stored
files = os.listdir(folder_test)
# Delete files in test folder
for file in files:
    if file.endswith(".txt"):
        file_path = os.path.join(folder_test, file)
        os.remove(file_path)

domains = ["corp", "equi", "htfl", "wind"]
langs   = ["en", "fr", "nl"]

domain   = domains[0]       # Selecting a domain
language = langs[0]
index_text = 59 if domain in ["corp", "wind"] else 60     # For extracting file name with text
index_term = 114 if domain in ["corp", "wind"] else 115   # For extracting file name with true terms

folder_path = "/content/ACTER/" + language + "/" + domain + "/annotated/texts_tokenised"
file_list = os.listdir(folder_path)

# Read txt files with text
for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
            f = open("/content/tests/text_" + file_path[57:index_text] + ".txt", 'w+')
            with open("/content/tests/text_" + file_path[57:index_text] + ".txt", 'w+') as f:
              f.write(file_content)

# Extract true terms from ACTER
ann_path = '/content/ACTER/'+ language + "/" + domain + "/annotated/annotations/sequential_annotations/iob_annotations/with_named_entities/"
ann_list = os.listdir(ann_path)

# Read files with true terms
true_terms = []
word = ''

for filename in ann_list:
    if filename.endswith('.tsv'):
        file_path = os.path.join(ann_path, filename)
        df = pd.read_table(file_path, sep = '\t', header = None)
        df.to_csv('/content/file.csv')

        for i in range(len(df)):
            if df.iloc[i,1] == 'B':
                true_terms.append(word)
                word = df.iloc[i,0]
            elif df.iloc[i,1] == 'I':
               word = word + " " + df.iloc[i,0]

        true_terms.append(word)
        true_terms.pop(0)
        f = open("/content/tests/term_" + file_path[112:index_term] + ".txt", 'w+')
        with open("/content/tests/term_" + file_path[112:index_term] + ".txt", 'w') as f:
            f.write(', '.join(true_terms))
        true_terms.clear()
        word = ''

"""
Reading txt files from the folder "test"
"""

folder_name ='/content/tests'
all_files = os.listdir(folder_name)   # Get a list of all files in the folder
file_tuples = []                      # Initialize empty lists to store the tuples

# Iterate through the files in the folder
for filename in all_files:
    if filename.startswith("text_"):              # Check if the file is a test file (starts with "text_")
        result_filename = "term_" + filename[5:]  # Construct the expected result filename by replacing "text_" with "term_"
        if result_filename in all_files:          # Check if the corresponding result file exists in the folder
            file_tuples.append((filename, result_filename))
file_tuples=sorted(file_tuples)

"""
Functions for term extraction
"""

def parse_candidates(text:str):
    infixes = [pattern for pattern in nlp.Defaults.infixes if "-" not in pattern]   # Remove the default rule that splits on hyphens
    infix_regex = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer

    doc = nlp(text)      # Process the text with spaCy

    # Extract MWEs (noun phrases) from the text
    mwe_list = []
    single_noun_list = dict()
    candidate_list = []

    for sent in doc.sents:
        sent_encode = model.encode(sent.text, convert_to_tensor = True).to(device)
        for chunk in sent.noun_chunks:
            is_candidate = False
            word_count = 0
            if len(chunk.text.split()) > 1:
                noun_appeared = False
                is_candidate = True
                cleared_candidate = ''
                for word in chunk:
                    if word.text == '-':
                        cleared_candidate += '-'
                        continue
                    #IGNORING
                    if word.pos_ in ['PUNCT', 'ADP', 'DET']:
                        continue
                    elif word.pos_ not in ['ADJ', 'PROPN', 'NOUN']:
                        is_candidate = False
                        break
                    elif word.pos_ in ['PROPN', 'NOUN']:
                        noun_appeared = True
                    elif not(not noun_appeared and word.pos_ == 'ADJ'):
                        is_candidate = False
                        break
                    cleared_candidate += word.text+' '
                    word_count += 1
            if is_candidate and word_count > 1:
                cleared_candidate = cleared_candidate.strip()
                candidate_list.append(CandidateMWE(cleared_candidate, chunk.root.text, sent.text, model.encode(cleared_candidate, convert_to_tensor = True).to(device), sent_encode))
            else:
                for word in chunk:
                    if word.pos_ in ['NOUN', 'PROPN']:
                        single_noun_list[word.text] =  CandidateW(word.text, word.lemma_, model.encode(word.text))

    return candidate_list, single_noun_list.values()

def parse_candidates_new(text:str):
    infixes = [pattern for pattern in nlp.Defaults.infixes if "-" not in pattern]        # Remove the default rule that splits on hyphens
    infix_regex = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    doc = nlp(text)  # Process the text with spaCy

    # Extract MWEs (noun phrases) from the text
    mwe_list = []
    single_noun_list = dict()
    candidate_list = []

    for sent in doc.sents:
        sent_encode = model.encode(sent.text, convert_to_tensor = True).to(device)
        temp_list = []
        for word in sent:
            if word.pos_ in ['ADJ', 'ADP', 'DET']:
                temp_list.append(word)
            elif word.pos_ in ['NOUN', 'PROPN']:
                single_noun_list[word.text] = CandidateW(word.text, word.lemma_, model.encode(word.text))
                temp_list.append(word)
                for i in range(1, len(temp_list)):
                    chunk = doc[temp_list[-i-1].i:temp_list[-1].i+1]
                    cleared_candidate = chunk.text
                    candidate_list.append(CandidateMWE(cleared_candidate, chunk.root.text, sent.text, model.encode(cleared_candidate, convert_to_tensor = True).to(device), sent_encode))
            else:
                temp_list = []

    return candidate_list, single_noun_list.values()

def dist(wi_encode, wj_encode):
    return util.pytorch_cos_sim(
        wi_encode,
        wj_encode
    )

def calculate_topic_score(expression_embedding, sentence_embedding)->float:
    """
    Calculate the topic score between a multiword expression and a sentence.

    Args:
        multiword_expression (str): The multiword expression.
        sentence (str): The sentence containing the expression.

    Returns:
        float: The topic score (cosine similarity) between the two embeddings.
    """
    # Load the distilbert-base-nli-mean-tokens
    # Encode the multiword expression and sentence into embeddings
    # expression_embedding = model.encode(multiword_expression, convert_to_tensor=True)
    # sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    similarity_score = util.pytorch_cos_sim(expression_embedding, sentence_embedding)        # Calculate cosine similarity between the two embeddings
    topic_score = similarity_score[0].item()                                                 # Extract the cosine similarity value from the tensor

    return topic_score


def calculate_specificity_score(mw:CandidateMWE, full_encode:Tensor)->float:
    """
    Calculate the specificity score (SP) between a multiword expression (mw) and a list of words/multiword expressions (w).

    Args:
        mw (str): The multiword expression.
        w (list of str): The list of words/multiword expressions in the context.

    Returns:
        float: The specificity score (SP).
    """
    # Load the distilbert-base-nli-mean-tokens model
    distances = dist(mw.self_encode,full_encode)      # Calculate distances between mw and each word/phrase in w
    specificity_score = distances.mean().item()       # Calculate the mean of the distances

    return specificity_score

def detect_mw_terms(candidate_list:List[CandidateMWE], TSP:float = 0.05, Ttopic:float = 0.1)->List[CandidateMWE]:
    full_encode = torch.stack([wi.self_encode for wi in candidate_list], dim = 0)
    temp_candidate = []
    for candidate in candidate_list:
        topic_score = calculate_topic_score(candidate.self_encode, candidate.sent_encode)
        sp_score = calculate_specificity_score(candidate, full_encode)
        if topic_score > Ttopic and sp_score > TSP:
            temp_candidate.append(candidate)

    return temp_candidate

def detect_single_noun_terms(term_mws:List[CandidateMWE], single_noun_list:List[CandidateW], subtoken_threshold:int = 4)->List[CandidateW]:
    term_nouns = []
    for candidate in single_noun_list:
        #Check if the lemma of the noun is the same as any of the heads of the multiword expressions.
        is_term = False
        lemma_is_head = False
        for term_mw in term_mws:
            if term_mw.head == candidate.lemma:
                is_term = True
                term_nouns.append(candidate)
                break
        if is_term:
            continue

        #segment the word using a subword-unit segmentation and a vocabulary trained over a large general purpose corpus.
        subtokens = tokenizer.tokenize(candidate.text)
        if len(subtokens) > subtoken_threshold:
            term_nouns.append(candidate)
    return term_nouns

"""
Calculate metrics
"""

def calculate_metrics(true_terms, extracted_terms):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def compare_sets(extracted_terms, true_terms):
    true_detections = extracted_terms.intersection(true_terms)
    false_gaps = true_terms.difference(extracted_terms)
    false_detections = extracted_terms.difference(true_terms)

    return true_detections, false_gaps, false_detections

def report(true_terms, extracted_terms):
    precision, recall, f1_score = calculate_metrics(true_terms, extracted_terms)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    true_detections, false_gaps, false_detections = compare_sets(extracted_terms, true_terms)
    #print("Истинные обнаружения:", true_detections)
    #print("Ложные пропуски:", false_gaps)
    #print("Ложные обнаружения:", false_detections)
    return precision, recall, f1_score, true_detections, false_gaps, false_detections

"""
Testing in Folder
"""

def test_file(text_file_name, term_file_name):
    print('____________________________________________________________________\n')
    print(f"Checking files: {text_file_name}, {term_file_name}")
    text=''
    with open(text_file_name) as text_file:
        text=text_file.read()
    candidate_list, single_noun_list=parse_candidates(text)
    term_mws=detect_mw_terms(candidate_list)
    term_nouns=detect_single_noun_terms(term_mws, single_noun_list)
    extracted_terms=set([txt.text.lower() for txt in term_mws+term_nouns])

    with open(term_file_name) as term_file:
        terms_str=term_file.read()
    true_terms=set(terms_str.lower().split(', '))

    precision, recall, f1_score, true_detections, false_gaps, false_detections = report(true_terms, extracted_terms)
    return true_terms, extracted_terms, precision, recall, f1_score, true_detections, false_gaps, false_detections

os.chdir(folder_name)
whole_true_terms = []
whole_extracted_terms = []
results = []
for text_file_name, term_file_name in file_tuples:
    true_terms, extracted_terms, precision, recall, f1_score, true_detections, false_gaps, false_detections = test_file(text_file_name, term_file_name)
    results.append((true_terms, extracted_terms, precision, recall, f1_score, true_detections, false_gaps, false_detections ))

    whole_true_terms += true_terms
    whole_extracted_terms += extracted_terms
print('_____________________TOTAL RESULT___________________________\n')
whole_true_terms = set(whole_true_terms)
whole_extracted_terms = set(whole_extracted_terms)
precision, recall, f1_score, true_detections, false_gaps, false_detections=report(whole_true_terms, whole_extracted_terms)
