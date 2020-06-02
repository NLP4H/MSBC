import tensorflow as tf
print(tf.__version__)
import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pickle
import pandas as pd
import numpy as np
from os import path
import argparse


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
    note = " ".join(lemmatized)
    
    return note

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
    note = " ".join(filtered)
    return note

def notes_to_sequences(notes, save_dir, max_seq_length = 1000, note_type = "neurology", train = True, split_type="none"):

    """
    Turn list of texts into sequence matrix
    Input: List of texts
    Output: Sequence matrix
    saved file: tokenizer & word index
    """ 
    print("Preprocessing")
    # Text level pre-processing
    texts = []
    if note_type == "neurology":
        for i in range(len(notes)):
            texts.append(preproc_neurology(notes[i]))
    if note_type == "radiology":
        for i in range(len(notes)):
            texts.append(preproc_radiology(notes[i]))
    
    
    # If training, create tokenizer and generate training data
    if train == True: 

        # Tokenization and pad sequence
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=max_seq_length)
        print('Shape of data tensor:', data.shape)
        
        if note_type == "neurology":
            # Save tokenizer and word index
            with open(path.join(save_dir,'tokenizer_neurology.pickle'), 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open(path.join(save_dir, 'word_index_neurology.pickle'), 'wb') as handle:
                pickle.dump(word_index, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        if note_type == "radiology":
            # Save tokenizer and word index
            with open(path.join(save_dir, 'tokenizer_radiology.pickle'), 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
            with open(path.join(save_dir, 'word_index_radiology.pickle'), 'wb') as handle:
                pickle.dump(word_index, handle, protocol = pickle.HIGHEST_PROTOCOL)
        return data, tokenizer, word_index
    
    print("Creating sequence data")
    # Generate validation data
    if train == False:

        if note_type == "radiology":
            if split_type == "val":
                # Load saved tokenizer
                with open(path.join(save_dir, "tokenizer_radiology.pickle"), "rb") as handle:
                    tokenizer = pickle.load(handle)
            if split_type == "test":
                # Load saved tokenizer
                with open(path.join(save_dir, "tokenizer_radiology.pickle"), "rb") as handle:
                    tokenizer = pickle.load(handle)
        
        if note_type == "neurology":
            if split_type == "val":
                # Load saved tokenizer
                with open(path.join(save_dir, "tokenizer_neurology.pickle"), "rb") as handle:
                    tokenizer = pickle.load(handle)
            if split_type == "test":
                # Load saved tokenizer
                with open(path.join(save_dir, "tokenizer_neurology.pickle"), "rb") as handle:
                    tokenizer = pickle.load(handle)
        
        # Pad sequences to text
        sequences = tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=max_seq_length)
        print('Shape of data tensor:', data.shape)
        
        return data

def load_data(train_dir, val_dir, test_dir, save_dir, note_type = "neurology", note_column = "text"):

    print("Loading training data ...")
    # Read train file
    df_train = pd.read_csv(train_dir)
    # Remove places where there's no note
    df_train = df_train.dropna(subset = [note_column])
    # Fill NA values in labels to -1
    df_train = df_train.fillna(-1)
    df_train = df_train[df_train.edss_19 != -1]
    # Reset Index
    df_train = df_train.reset_index()
    
    print("Loading validation data ...")
    # Read validation file
    df_val = pd.read_csv(val_dir)
    # Remove places where there's no note
    df_val = df_val.dropna(subset = [note_column])
    # Fill NA values in labels to -1
    df_val = df_val.fillna(-1)
    df_val = df_val[df_val.edss_19 != -1]
    # Reset Index
    df_val = df_val.reset_index()

    print("Loading testing data ...")
    # Read validation file
    df_test = pd.read_csv(test_dir)
    # Remove places where there's no note
    df_test = df_test.dropna(subset = [note_column])
    # Fill NA values in labels to -1
    df_test = df_test.fillna(-1)
    df_test = df_test[df_test.edss_19 != -1]
    # Reset Index
    df_test = df_test.reset_index()
    
    if note_type == "neurology": 
        
        print("Transfer notes to sequence data")
        X_train, _, _ = notes_to_sequences(list(df_train[note_column]), save_dir, note_type = "neurology", train = True)
        X_val = notes_to_sequences(list(df_val[note_column]), save_dir, note_type = "neurology", train = False,split_type="val")
        X_test = notes_to_sequences(list(df_test[note_column]), save_dir, note_type = "neurology", train = False, split_type="test")

        print("Saving files")
        # Save data
        os.chdir(save_dir)
        np.savez("training_data_neurology.npz", X_train, X_val, X_test)
        return X_train, X_val, X_test

    if note_type == "radiology":
        
        print("Transfer notes to sequence data")
        X_train, _, _ = notes_to_sequences(list(df_train[note_column]), save_dir, note_type = "radiology", train = True)
        X_val = notes_to_sequences(list(df_val[note_column]), save_dir, note_type = "radiology", train = False, split_type="val")
        X_test = notes_to_sequences(list(df_test[note_column]), save_dir, note_type = "radiology", train = False, split_type="test")
        
        print("Saving files")
        # Save data
        os.chdir(save_dir)
        np.savez("training_data_radiology.npz", X_train,  X_val, X_test)
        return X_train, X_val

def prep_data(save_dir_intermediate_data, train_dir, val_dir, test_dir, mri_load, neurology_load):
    print("running prepare_data.py")
    save_dir = save_dir_intermediate_data

    # ---------------------------------------- NEUROLOGY ---------------------------------------------------------------------------
    if(neurology_load):
        # Save data as intermediate files
        print("Preparing neurology notes")
        train_dir = train_dir
        val_dir = val_dir
        test_dir = test_dir
        X_train, X_val, X_test = load_data(train_dir, val_dir, test_dir, save_dir)
    
    # ---------------------------------------- RADIOLOGY ---------------------------------------------------------------------------
    if(mri_load):
        print("Preparaing radiology notes")
        train_dir = train_dir
        val_dir = val_dir
        X_train, X_val = load_data(train_dir, val_dir, save_dir, note_type = "radiology", note_column="result")
    return
