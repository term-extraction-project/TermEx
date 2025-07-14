! git clone https://github.com/term-extraction-project/Matcha.git
import os
import pandas as pd
import csv

#################### MATCHA ####################################
# Connecting the Match case
path_to_corpus="/content/" #write  path to corpus

langs=["en","kaz"]               
domains=["block","material_sci"]

language=langs[1]                # Language selection    set kazakh , since this version is for the Kazakh language
domain=domains[0]                # domain selection

folder_path=path_to_corpus+"Matcha/"+language+"/"+domain+"/annotated/texts"          # Path to domain texts in the corresponding language
files_name=[]
texts=dict()
true_terms_by_doc=dict()
all_texts=""

file_list = os.listdir(folder_path)

for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            text = file.read().replace("  ", " ").lower().replace("- ","-")
            all_texts+=text+". "

            texts["text_"+file_path[-6:]] = text
            files_name.append("text_"+file_path[-6:])

all_true_terms=[]

ann_path = path_to_corpus+"/Matcha/"+language+"/"+domain+"/annotated/annotations/unique_annotation_lists/"+domain+"_"+language+"_terms.csv"    #  Path to domain terms in the corresponding language,    extract from csv file

df = pd.read_csv(ann_path, delimiter=';')
data_list = df.values.tolist()
all_true_terms=[i[0].lower() for i in data_list]

print('True terms all: ', len(set(all_true_terms)))

#################### ------- ####################################

! pip install stanza     # Download the Kazakh model for preprocessing text
import stanza
stanza.download('kk')
nlp = stanza.Pipeline('kk')   # Initialize the Kazakh pipeline

from torch import Tensor
!!pip install torch

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!!pip install sentence-transformers

# Load the language model
# Remove hyphen from the default infix patterns
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Alibaba-NLP/gte-multilingual-base',trust_remote_code=True)

from sentence_transformers import SentenceTransformer, util
from typing import List
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

from collections import namedtuple
CandidateMWE = namedtuple('CandidateMWE',['text','head', 'sentence','self_encode', 'sent_encode'])
CandidateW=namedtuple('CandidateW',['text','lemma', 'self_encode' ])

all_texts_lemms=" ".join([word.lemma for sent in nlp(all_texts).sentences for word in sent.words])

!git clone https://github.com/term-extraction-project/multi_word_expressions.git
import sys
sys.path.append('/content/multi_word_expressions/extractors')  #path to file with extractor

from kazakh import KazakhPhraseExtractor

characters="аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя 0123456789'-()’abcdefghijklmnopqrstuvwxyz"       # valid character values

def parse_candidates(text:str):
        # Remove the default rule that splits on hyphens
    # Process the text with spaCy
    doc = nlp(text)

    # Extract MWEs (noun phrases) from the text
    mwe_list = []
    single_noun_list=dict()
    candidate_list=[]
    temp_noun=[]
    for sent in doc.sentences:
        sent_encode=model.encode(sent.text, convert_to_tensor=True).to(device)
        extractor = KazakhPhraseExtractor(
                        text=sent,
                        #stop_words=custom_stop_words_en,   # Пользовательские стоп-слова, по умолчанию установлены
                        #list_seq=custom_pos_patterns_en,   # Пользовательские POS-шаблоны, по умолчанию установлены
                        cohision_filter=True,               # Фильтрация по когезии
                        additional_text=all_texts_lemms,    # Дополнительный текст (если требуется)
                        f_raw_sc=2,                         # Частотный фильтр для сырого текста
                        f_req_sc=1)                         # Частотный фильтр для отобранных кандидатов
        candidates = extractor.extract_phrases()
        candidates=list(set(candidates))

        for chunk in candidates:
                if len(set(chunk).intersection(set(characters)))==len(set(chunk)):
                    head=[word.text for word in nlp(chunk).sentences[0].words if word.head==0][-1]
                    candidate_list.append(CandidateMWE(chunk, head, sent.text, model.encode(chunk, convert_to_tensor=True).to(device), sent_encode))
                # print(f'Added candicate expression: {cleared_candidate}')

        for word in sent.words:             # extract single noun camdidates 
                    if word.upos in ['NOUN', 'PROPN']:
                       if len(set(word.text).intersection(set(characters)))==len(set(word.text)) and word.text not in temp_noun:
                           single_noun_list[word.text]=  CandidateW(word.text, word.lemma, model.encode(word.text))
                           temp_noun.append(word.text)

    return list(set(candidate_list)), single_noun_list.values()

def dist(wi_encode, wj_encode):  
    return util.pytorch_cos_sim(
        wi_encode,
        wj_encode
    )
def calculate_topic_score(expression_embedding, sentence_embedding)->float:           # calculate topic score
    """
    Calculate the topic score between a multiword expression and a sentence.

    Args:
        multiword_expression (str): The multiword expression.
        sentence (str): The sentence containing the expression.

    Returns:
        float: The topic score (cosine similarity) between the two embeddings.
    """
    # Load the distilbert-base-nli-mean-tokens model

    # Encode the multiword expression and sentence into embeddings
    # expression_embedding = model.encode(multiword_expression, convert_to_tensor=True)
    # sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # Calculate cosine similarity between the two embeddings
    similarity_score = util.pytorch_cos_sim(expression_embedding, sentence_embedding)

    # Extract the cosine similarity value from the tensor
    topic_score = similarity_score[0].item()

    return topic_score


def calculate_specificity_score(mw:CandidateMWE, full_encode:Tensor)->float:              # calculate specific score
    """
    Calculate the specificity score (SP) between a multiword expression (mw) and a list of words/multiword expressions (w).

    Args:
        mw (str): The multiword expression.
        w (list of str): The list of words/multiword expressions in the context.

    Returns:
        float: The specificity score (SP).
    """
    # Load the distilbert-base-nli-mean-tokens model
    # Calculate distances between mw and each word/phrase in w
    distances = dist(mw.self_encode,full_encode)

    # Calculate the mean of the distances
    specificity_score = distances.mean().item()


    return specificity_score


def detect_mw_terms(candidate_list:List[CandidateMWE], TSP:float = 0.15, Ttopic:float = 0.6)->List[CandidateMWE]:        # filter phrases by topic and specific score

    full_encode= torch.stack([wi.self_encode for wi in candidate_list], dim=0)

    temp_candidate = []
    for candidate in candidate_list:
        topic_score = calculate_topic_score(candidate.self_encode, candidate.sent_encode)
        sp_score=calculate_specificity_score(candidate, full_encode)
        if topic_score > Ttopic and sp_score > TSP:
            temp_candidate.append(candidate)
            # print(f'Added "{candidate.text}" topic score {topic_score}, specifity score {sp_score}')

    return temp_candidate

def detect_single_noun_terms(term_mws:List[CandidateMWE], single_noun_list:List[CandidateW], subtoken_threshold:int=4)->List[CandidateW]:        # filter single nouns
    term_nouns=[]
    for candidate in single_noun_list:
        #Check if the lemma of the noun is the same as any of the heads of the multiword expressions.
        is_term=False
        lemma_is_head=False
        for term_mw in term_mws:
            if term_mw.head==candidate.lemma:
                is_term=True
                term_nouns.append(candidate)
                # print(f'"{candidate.text}" is added by lemma: "{candidate.lemma}" is head of "{term_mw}"')
                break
        if is_term:
            continue
        #segment the word using a subword-unit segmentation and a vocabulary trained over a large general purpose corpus.
        subtokens=tokenizer.tokenize(candidate.text)
        if len(subtokens)>subtoken_threshold:
            term_nouns.append(candidate)
            # print(f'{candidate.text} is added by subtokens count: {len(subtokens)}')
    return term_nouns

def test_file(text):        # monitoring the stage of term extraction process

    print('_____________________________FIRST STEP_________________________________________')
    candidate_list, single_noun_list=parse_candidates(text)
    print(len(set(candidate_list)))
    print('_____________________________SECOND STEP___________________________________________________')
    term_mws=detect_mw_terms(candidate_list)
    print('_____________________________THIRD STEP______________________________________________________________')
    term_nouns=detect_single_noun_terms(term_mws, single_noun_list)
    print('____________________________________TESTING__________________________________________________________________')
    extracted_terms=set([txt.text.lower() for txt in term_mws+term_nouns])
    print(len(set(term_mws)))
    print(set(extracted_terms))

    return extracted_terms

def calculate_metrics(true_terms, extracted_terms):     # calculate metrics precision, recall, f1 score
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
    # Расчет метрик
    precision, recall, f1_score = calculate_metrics(true_terms, extracted_terms)

    print("Precision:", precision) #Точность
    print("Recall:", recall) #Полнота
    print("F1 Score:", f1_score)
    # Выводит списки истинных обнаружений, ложных пропуски и ложных обнаружений
    true_detections, false_gaps, false_detections = compare_sets(extracted_terms, true_terms)

    print("Истинные обнаружения:", true_detections)
    print("Ложные пропуски:", false_gaps)
    print("Ложные обнаружения:", false_detections)
    return precision, recall, f1_score, true_detections, false_gaps, false_detections



f1_score_set=[]
whole_extracted_terms=[]
results=[]
kk=[]
stop_words=['басқалай', 'сенің', 'бірнеше', 'қазіргі', 'егерде', 'соншалықты', 'жылдардың', 'осы', 'неше', 'себебі', 'секілді', 'осылайша', 'бірақ та', 'қайда', 'кездесі', 'ешқандай', 'қашан', 'жоғары', 'өте', 'бірде', 'өз', 'де',
            'анау', 'сияқты', 'біреу', 'қалай', 'қайталай', 'егер', 'мынау', 'олар', 'арасында', 'сен', 'ешқашан', 'бірдеңе', 'не', 'оң', 'тағы да', 'бар', 'дегенмен', 'ешкім', 'қайта', 'алайда', 'қазір', 'да', 'барлық', 'әркім',
            'бәрі', 'байлынысты', 'жоқ', 'жиі', 'сондықтан да', 'біз', 'кейін', 'сол', 'соңғы', 'мүмкін', 'олай болса', 'айналасында', 'төмен', 'ішінде', 'болуы мүмкін', 'қаншалықты', 'бәрібір', 'соңында', 'дейін', 'сіз', 'осында',
            'туралы', 'олардікі', 'әрдайым', 'қандай', 'қалайша', 'мен', 'бірге', 'осылай', 'оның', 'ал', 'болатын', 'әр', 'және', 'алыс', 'әрі', 'кез-келген', 'сондықтан', 'кейбіреулер', 'бұрын', 'неге', 'кейінірек', 'арнайы', 'басқа',
            'байланысты', 'ертең', 'ғана', 'кеше', 'сіздің', 'сонда', 'кім', 'тек', 'әлдеқайда', 'жылы', 'тамаша', 'сирек', 'барлығы', 'бірақ', 'кезде', 'бастап', 'бұл', 'қай жерде', 'кезінде', 'үшін', 'ол', 'болып табылады', 'сондай',
            'біздің', 'мұнда', 'менің', 'кейде', 'арқылы', 'болды', 'тағы', 'жылдың', 'сыртында', 'әрқашан', 'жақын', 'олардың', 'онда', 'сондай-ақ', 'қанша', 'біздікі', 'бәріміз', 'бүгін', 'ештеңе', 'көптеген']

for file_name in files_name:
    print('____________________________________________________________________\n')
    print(f"checking file: {file_name}")

    extracted_terms=test_file(texts[file_name])
    results.append(extracted_terms)
    extracted_terms=[i for i in extracted_terms if i[0]!="-" and  i[-1]!="-"]
    extracted_terms=set(extracted_terms) - set([i for i in extracted_terms for w in stop_words if w in i.split(" ")])   # cleaning phrases with stop words

    print('____________________________________________________________________\n')

    true_terms_per_text=[i for i in set(all_true_terms) if i in texts[file_name]]

    precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(true_terms_per_text), set(extracted_terms))
    whole_extracted_terms+=extracted_terms
print('___________________________TOTAL_________________________________________\n')
whole_extracted_terms=set(whole_extracted_terms)
print('total result')
print(len(set(whole_extracted_terms)))
print(set(whole_extracted_terms))

precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(all_true_terms), whole_extracted_terms)

# lemmatization extracted terms and true terms for check the efficiency of UA1
mwe=[i for i in whole_extracted_terms if len(i.split(" "))>1]+[i for i in whole_extracted_terms if len(i.split("-"))>1]
uni=set(whole_extracted_terms)-set(mwe)

tt_lemma_u=[i.lemma for tt in set(true_terms_uni)  for s in nlp(tt).sentences  for i in s.words]
ext_lemma_u=[i.lemma for tt in set(uni)for s in nlp(tt).sentences  for i in s.words]
tt_lemma_m=[]
for i in set(true_terms_mwe):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  tt_lemma_m.append(" ".join(temp))

ext_lemma_m=[]
for i in set(mwe):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  ext_lemma_m.append(" ".join(temp))

# print results for lematize true terms and extracted terms
precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(tt_lemma_u+tt_lemma_m), set(ext_lemma_m+ext_lemma_u))
