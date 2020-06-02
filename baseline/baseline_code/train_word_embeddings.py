"""
This file trains the Word2Vec embeddings for Neurology and Radiology notes
And save them as .pickle file

"""

import os
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import re
import string

# gensim
from gensim.models import Word2Vec, keyedvectors

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import prepare_data
from os import path



def preproc_radiology(note):
    
    """
    Pre-processing for MRI radiology notes
    Input: one piece of radiology note
    Output: pre-processed piece of radiology note

    """
    
    # Stop words
    stop_words = set(stopwords.words('english'))
    # Punctuations except '.'
    punc = string.punctuation.replace(".", "")
    translator = str.maketrans('', '', punc)
    
    # --------------- Remove Irrelevant Info ---------------------------
    p1 = re.compile(r"\(MRI\).*(Accession|A#)\:\s+\d{7}")
    p2 = re.compile(r"[A-Z' \.]*M\.D\.\s+\(Staff\)|CC\:\s+[A-Z' \.]*")
    note = re.sub(p1, r"", note)
    note = re.sub(p2, r"", note)
    note = re.sub(r"\s+", " ", note)
    
    # ----------------- Text Level Preproc ------------------------------
    note = note.lower()
    note = note.replace("-", " ")
    note = note.translate(translator)
    # Tokenization
    tokens = word_tokenize(note) 
    # Remove stop words
    filtered = [w for w in tokens if not w in stop_words]
    # lemmatization
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
    # note = " ".join(lemmatized)
    
    return lemmatized

def preproc_neurology(note):
    
    """
    Pre-processing for neurology notes
    Input: one piece of neurology note
    Output: pre-processed piece of neurology note
    
    """ 
    # ----------------- Remove irrelevant info ---------------------
    # Header
    # Diagnosis...Dear
    p = re.compile(r'(?:procedure|DIAGNOSIS|diagnoses|diagnosis\(es\)|clinical history|history of present illness).*(?:Dear|To whom it may concern).*', re.IGNORECASE)
    # Dear
    p1 = re.compile(r'(?=Dear|To whom it may concern).*', re.IGNORECASE)
    # Footer
    p2 = re.compile(r'(Yours sincerely|sincerely|With kind regards|kind regards|Best regards).*', re.IGNORECASE)
    p3 = re.compile(r'D:.*(A|P)?\sT\:.*')
    # Signature
    p4 = re.compile(r'(Electronically Signed by\s)?([A-Z][a-z]+|[A-Z].)+\s[A-Z][a-z]+,\s(MD|M.D.)')
    p5 = re.compile(r'(Electronically Signed by\s)?[A-z][a-z]+\s[A-Z][a-zA-Z]*(-[A-Z][a-zA-Z]*),\s(MD|M.D.)')

    # Renmove header & footer
    # Header
    if len(re.findall(p, note)) > 0:
        note = re.findall(p, note)[0]
    elif len(re.findall(p1, note)) > 0: 
        note = re.findall(p1, note)[0]
    # Footer
    if len(re.findall(p2, note)) > 0 or \
        len(re.findall(p3, note)) > 0 or \
            len(re.findall(p4, note)) > 0 or \
                len(re.findall(p5, note)) > 0:
                note = re.sub(p2, r'', note)
                note = re.sub(p3, r'', note)
                note = re.sub(p4, r'', note)
                note = re.sub(p5, r'', note)

    # Patterns
    # Telephone & Fax
    p1 = re.compile(r'(phone:|fax:|tel:|#|\s)(\s?)\(?\d{3}\)?(-?|\s?)\d{3}(-?|\s?)\d{4}', re.IGNORECASE)
    # MRN
    p2 = re.compile(r'(\(?MRN\)?:?\s?\d{3}(-?)\d{4})|\d{5,6,7}|(MRN\s#\d{5,6,7})|MRN\s\"\d{8}\"')
    # D.O.B
    p3 = re.compile(r'((D.O.B.|DOB)?\s?:?\s?)(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\/\d{1,2}\/\d{4}|[A-Za-z]{3}-\d{1,2}-\d{4}|\d{4}(-|\/)[A-Za-z]{3}(-|\/)\d{1,2}|[A-Z][a-z]+\s\d{1,2}, \d{4}|\d{1,2}\/\d{1,2}\/\d{2,4})', re.IGNORECASE)
    # Other dates
    p4 = re.compile(r'\d{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December),?\s\d{4}')
    # Time
    p5 = re.compile(r'(\d{2}:\d{2})|(\d{2}:\d{2}:\d{2})')
    # Address
    p6 = re.compile(r'\d{1,3}.?\d{0,3}\s[a-zA-Z]{2,30}\s(Street|St.|Avenue|Ave.)') # Street No. and Name
    p7 = re.compile(r'(Toronto,)?\sON\s[A-Z]\d[A-Z]\s?\d[A-Z]\d', re.IGNORECASE) # "City & Postal code"
    # cc: and make sure it's at the end of the sentence
    p8 = re.compile(r'cc:.{0,1000}$')
           
    # Phone/Fax
    note = re.sub(p1, r'', note)
    # MRN#
    note = re.sub(p2, r'', note)
    note = re.sub(r'Patient Identifier', r'', note)
    # DOB
    note = re.sub(p3, r'', note)
    note = re.sub(r'Date of Birth \(DOB\) \((MON dd, yyyy|yyyy-mm-dd)\)|Date of Birth \(DOB\)|DATE OF BIRTH', r'', note)
    # Other dates
    note = re.sub(p4, r'', note)
    # Time
    note = re.sub(p5, r'', note)
    # Address info
    note = re.sub(p6, r'', note)
    note = re.sub(p7, r'', note)

    # Check whether 'cc:' is in the footer & remove everything after 'cc:'
    if len(re.findall(p8, note)) > 0:
            note = re.sub(r'cc:.*', r'', note)
    if len(re.findall(r'CC:.{0,1000}$', note)) > 0:
            note = re.sub(r'CC:.*', r'', note)


    # -----------------Text Level Preproc--------------
    note = note.lower()
    # Remove extra spaces
    note = re.sub(r"\s+", r" ", note)
    # Replace "-" with space (e.g. for phrases like "40-year-old")
    note = re.sub(r"\-", r" ", note)
    # Stop words
    stop_words = set(stopwords.words('english'))
    # Punctuations except '.'
    punc = string.punctuation.replace(".", "")
    translator = str.maketrans('', '', punc)
    # Remove irrelevant punctuations
    note = note.translate(translator)
    # Tokenization
    tokens = word_tokenize(note) 
    # Remove stop words
    filtered = [w for w in tokens if not w in stop_words]
    # note = " ".join(filtered)
    return filtered


def train_word_embeddings(save_dir, text_processed_csv_path, neurology_load, mri_load):
    

    save_dir = save_dir

    # --------------------------------------------- NEUROLOGY -------------------------------------------------
    if(neurology_load):
        # Load all available notes
        print("Neurology notes")
        print("Load data...")

        df = pd.read_csv(text_processed_csv_path)

        # Remove irrelevant notes
        # Notes that should be excluded: Operative Summary, D/C Summary, eAdmit, Input Consult, ER Dpt Consult
        #                                Transfer Summary, Lithotripsy Rep, D/C Summ-Letter, Delivery Summary
        include_list = ['Amb. Consult',
                        'AmbProgressSumm',
                        'MartinFamArthrit',
                        'Mental Health',
                        'Office Consult',
                        'OfficeProgressSu',
                        'Other']
        df = df.loc[df['FindingAbbreviation'].isin(include_list)]
        # Print summary
        print("Data Summary: ")
        print(df["FindingAbbreviation"].value_counts())

        # Text Cleaning
        print("Preprocessing ...")
        notes = df["text"].tolist()
        texts = []
        for i in range(df.shape[0]):
            texts.append(preproc_neurology(notes[i]))
        print("Preprocessing results: ")
        print(texts[0])

        # Train Word2Vec model
        print("Training Word2Vec ...")
        model = Word2Vec(texts, size=200, window=10, min_count=2, workers=10, iter=10)

        # Test word similarity
        w1 = 'progression'
        print("Most similar word to: ", w1)
        print(model.wv.most_similar(positive = w1))

        # Save word vectors to dictionary
        print("Saving word vectors ...")
        vocab = []
        for word in model.wv.vocab.keys():
            vocab.append(word)
        word_vectors = model.wv
        embeddings_index = {}
        for word in vocab:
            embeddings_index[word] = word_vectors[word]

        # Save into dictionary
        with open(path.join(save_dir, 'ms_word2vec.pickle'), 'wb') as handle:
            pickle.dump(embeddings_index, handle, protocol = pickle.HIGHEST_PROTOCOL)
    

    
    # -------------------------------------------------- RADIOLOGY---------------------------------------
    if(mri_load):
        print("Radiology notes")
        print("Load data...")
        df_raw = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project_new/data/mri_data/mri_notes.csv")
        
        # Text cleaning
        print("Preprocessing ...")
        notes = df_raw["result"].tolist()
        texts = []
        for i in range(len(notes)):
            texts.append(preproc_radiology(notes[i]))
        print("Preprocessing results: ")
        print(texts[0])
        
        # Train Word2Vec model
        print("Training Word2Vec ...")
        model = Word2Vec(texts, size=200, window=10, min_count=2, workers=10, iter=10)


        # Test word similarity
        w1 = 'enhance'
        print("Most similar word to: ", w1)
        print(model.wv.most_similar(positive = w1))

        # Save word vectors to dictionary
        print("Saving word vectors ...")
        vocab = []
        for word in model.wv.vocab.keys():
            vocab.append(word)
        word_vectors = model.wv
        embeddings_index = {}
        for word in vocab:
            embeddings_index[word] = word_vectors[word]
        
        # Save into dictionary
        with open(path.join(save_dir, 'mri_word2vec.pickle'), 'wb') as handle:
            pickle.dump(embeddings_index, handle, protocol = pickle.HIGHEST_PROTOCOL)