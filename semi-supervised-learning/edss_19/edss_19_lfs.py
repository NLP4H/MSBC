# Snorkel labelling function for edss_19
# tutorial for reference: https://www.snorkel.org/use-cases/01-spam-tutorial

import pandas as pd
import numpy as np
import os

# will need to use GPU if predicting with cnn - uncomment line below
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import sklearn
from sklearn import metrics
import snorkel
from snorkel.labeling import labeling_function
import subprocess
import pdb

@labeling_function()
def LF_edss_19_int(df_row):
    """
    Look for EDSS score in int format
    Input: single row of df
    Output: EDSS class (0-18)
    
    """
        
    note = df_row.text
    p = re.compile(r"edss", re.IGNORECASE)
    p_int = re.compile(r"(?:\s|\,|\=)(?:[0-9])(?:\.|\,|\;|\-|\s+|\))")
    p_dec = re.compile(r"\d\.\d")

    score = -1      # default to -1 if not found
    sentences = sent_tokenize(note)
    for sent in sentences:
        edss_mentions = re.search(p, sent)

        # Find sentence with "EDSS"
        if edss_mentions != None:
            filtered_sentence = sent[edss_mentions.end():]

            # if score mentioned in int form
            if len(re.findall(p_int, filtered_sentence)) > 0:
                # if number is not decimal
                if len(re.findall(p_dec, filtered_sentence)) == 0:
                    # get first number that is mentioned after the edss mention
                    number = re.findall(p_int, filtered_sentence)[0]
                    score = float(re.sub(r"\s|\.|\=|\;|\-|\,|\)", r"", number))
                    break

    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1

    # Labels converted to int categories
    label_dict = {0.0:0,
            1.0:1,
            1.5:2,
            2.0:3,
            2.5:4,
            3.0:5,
            3.5:6,
            4.0:7,
            4.5:8,
            5.0:9,
            5.5:10,
            6.0:11,
            6.5:12,
            7.0:13,
            7.5:14,
            8.0:15,
            8.5:16,
            9.0:17,
            9.5:18,
            -1:-1}
        
    return label_dict[score]

@labeling_function()
def LF_edss_19_dec(df_row):
    """
    Look for EDSS score in decimal
    Input: single row of df
    Output: EDSS class (0-18)
    
    """
    note = df_row.text
    p = re.compile(r"edss", re.IGNORECASE) 
    p_dec = re.compile(r"\d\.\d")

    score = -1      # default to -1 if not found
    sentences = sent_tokenize(note)
    for sent in sentences:
        edss_mentions = re.search(p, sent)

        # Find sentence with "EDSS"
        if edss_mentions != None:
            filtered_sentence = sent[edss_mentions.end():]

            # if score mentioned in decimal form
            if len(re.findall(p_dec, filtered_sentence)) > 0:
                # get first number that is mentioned after the edss mention
                score = float(re.findall(p_dec, filtered_sentence)[0])
                break

    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1

    # Labels converted to int categories
    label_dict = {0.0:0,
            1.0:1,
            1.5:2,
            2.0:3,
            2.5:4,
            3.0:5,
            3.5:6,
            4.0:7,
            4.5:8,
            5.0:9,
            5.5:10,
            6.0:11,
            6.5:12,
            7.0:13,
            7.5:14,
            8.0:15,
            8.5:16,
            9.0:17,
            9.5:18,
            -1:-1}
        
    return label_dict[score]

@labeling_function()
def LF_edss_19_word(df_row):
    """
    Look for EDSS score in word formats
    Input: single row of df
    Output: EDSS class (0-18)
    """
    
    note = df_row.text
    p = re.compile(r"edss", re.IGNORECASE)   
    p_word = re.compile(r"zero|one|two|three|four|five|six|seven|eight|nine", re.IGNORECASE)     # looking for score in word form

    word_dict = {
    "zero":0.0,
    "one":1.0,
    "two":2.0,
    "three":3.0,
    "four":4.0,
    "five":5.0,
    "six":6.0,
    "seven":7.0,
    "eight":8.0,
    "nine":9.0
    }

    score = -1      # default to -1 if not found
    sentences = sent_tokenize(note)
    for sent in sentences:
        edss_mentions = re.search(p, sent)

        # Find sentence with "EDSS"
        if edss_mentions != None:
            filtered_sentence = sent[edss_mentions.end():]

            # find score mentioned in word form
            if len(re.findall(p_word, filtered_sentence)) > 0:
                # get first number that is mentioned after the edss mention
                score = float(word_dict[re.findall(p_word, filtered_sentence)[0].lower()])
                break

    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1

    # Labels converted to int categories
    label_dict = {0.0:0,
            1.0:1,
            1.5:2,
            2.0:3,
            2.5:4,
            3.0:5,
            3.5:6,
            4.0:7,
            4.5:8,
            5.0:9,
            5.5:10,
            6.0:11,
            6.5:12,
            7.0:13,
            7.5:14,
            8.0:15,
            8.5:16,
            9.0:17,
            9.5:18,
            -1:-1}
        
    return label_dict[score]

@labeling_function()
def LF_edss_19_roman(df_row):
    """
    Look for EDSS score in roman numeral format
    Input: single row of df
    Output: EDSS class (0-18)
    
    """

    note = df_row.text
    p = re.compile(r"edss", re.IGNORECASE)   
    p_roman = re.compile(r"(?:\s|\=|\,|[ ]\b)(IX|IV|V?I{0,3})\b", re.IGNORECASE)

    roman_dict = {
    "i":1.0,
    "ii":2.0,
    "iii":3.0,
    "iv":4.0,
    "v":5.0,
    "vi":6.0,
    "vii":7.0,
    "viii":8.0,
    "ix":9.0,
    }

    score = -1      # default to -1 if not found
    sentences = sent_tokenize(note)
    for sent in sentences:
        edss_mentions = re.search(p, sent)

        # Find sentence with "EDSS"
        if edss_mentions != None:
            filtered_sentence = sent[edss_mentions.end():]
            roman = re.findall(p_roman, filtered_sentence)
            roman_valid = [s for s in roman if s != ""]

            # find score mentioned in roman numeral form
            if len(roman_valid) > 0:
                # get first number that is mentioned after the edss mention
                score = float(roman_dict[roman_valid[0].lower()])
                break

    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1

    # Labels converted to int categories
    label_dict = {0.0:0,
            1.0:1,
            1.5:2,
            2.0:3,
            2.5:4,
            3.0:5,
            3.5:6,
            4.0:7,
            4.5:8,
            5.0:9,
            5.5:10,
            6.0:11,
            6.5:12,
            7.0:13,
            7.5:14,
            8.0:15,
            8.5:16,
            9.0:17,
            9.5:18,
            -1:-1}
        
    return label_dict[score]

@labeling_function()
def LF_edss_cnn_word2vec(df_path):
    """
    Predict edss_19 using a cnn model. CNN model uses trained Word2Vec embeddings. 
    Input: 
        df_path to evaluate
    Output: EDSS class (0-18) for entire df
    """
    # predict
    y_pred = subprocess.check_output(["master_path/miniconda3/bin/python","predict_cnn.py"] + [df_path])
    # above returns in weird byte format : b'class' -> thus process into string and extract class into numpy list
    y_pred = str(y_pred).split("'")
    #get 'class'
    y_pred = y_pred[1]
    #get into list format
    y_pred = y_pred.strip()
    y_pred = [int(s) for s in y_pred.split()]

    return np.array(y_pred) 


@labelling_function()
def LF_edss_cnn_BERT(df_row):
    """
    Take our offline predictions from our CNN model based on MS-BERT embeddings.
    Input:
        single row of df
        column name containging offline predictions
    """
    column = "predict_edss_19_allen_cnn"
    y_pred = df_row[column]

    return y_pred if y_pred != "NA" and y_pred is not None else ABSTAIN

  @labeling_function()
def LF_edss_tfidf_logreg(df_path):
    """
    Predict edss_19 using a logreg model
    Input: 
        df_path to evaluate
    Output: EDSS class (0-18) for entire df
    """
    # predict
    model_type = "log_reg_baseline"
    y_pred = subprocess.check_output(["master_path/miniconda3/bin/python","predict_tfidf.py"] + [df_path] + [model_type])
    # above returns in weird byte format : b'class' -> thus process into string and extract class into numpy list
    y_pred = str(y_pred).split("'")
    #get 'class'
    y_pred = y_pred[1]
    #get into list format
    y_pred = y_pred.strip()
    y_pred = [int(s) for s in y_pred.split()]

    return np.array(y_pred)

@labeling_function()
def LF_edss_tfidf_lda(df_path):
    """
    Predict edss_19 using a logreg model
    Input: 
        df_path to evaluate
    Output: EDSS class (0-18) for entire df
    """
    # predict
    model_type = "lda"
    y_pred = subprocess.check_output(["master_path/miniconda3/bin/python","predict_tfidf.py"] + [df_path] + [model_type])
    # above returns in weird byte format : b'class' -> thus process into string and extract class into numpy list
    y_pred = str(y_pred).split("'")
    #get 'class'
    y_pred = y_pred[1]
    #get into list format
    y_pred = y_pred.strip()
    y_pred = [int(s) for s in y_pred.split()]

    return np.array(y_pred)

@labeling_function()
def LF_edss_tfidf_svc_rbf(df_path):
    """
    Predict edss_19 using a logreg model
    Input: 
        df_path to evaluate
    Output: EDSS class (0-18) for entire df
    """
    # predict
    model_type = "svc_rbf"
    y_pred = subprocess.check_output(["master_path/miniconda3/bin/python","predict_tfidf.py"] + [df_path] + [model_type])
    # above returns in weird byte format : b'class' -> thus process into string and extract class into numpy list
    y_pred = str(y_pred).split("'")
    #get 'class'
    y_pred = y_pred[1]
    #get into list format
    y_pred = y_pred.strip()
    y_pred = [int(s) for s in y_pred.split()]

    return np.array(y_pred)

@labeling_function()
def LF_edss_tfidf_linear_svc(df_path):
    """
    Predict edss_19 using a logreg model
    Input: 
        df_path to evaluate
    Output: EDSS class (0-18) for entire df
    """
    # predict
    model_type = "linear_svc"
    y_pred = subprocess.check_output(["master_path/miniconda3/bin/python","predict_tfidf.py"] + [df_path] + [model_type])
    # above returns in weird byte format : b'class' -> thus process into string and extract class into numpy list
    y_pred = str(y_pred).split("'")
    #get 'class'
    y_pred = y_pred[1]
    #get into list format
    y_pred = y_pred.strip()
    y_pred = [int(s) for s in y_pred.split()]

    return np.array(y_pred)

def get_edss_19_lfs():
    return [LF_edss_19_int, LF_edss_19_dec, LF_edss_19_word, LF_edss_19_roman, LF_edss_cnn_word2vec, LF_edss_cnn_BERT] 
        #FOUND BETTER PERFORMANCE IN SNORKEL WITH JUST RULE BASED + CNN MODULES. REMOVING LF'S BELOW FOR NOW
        #, LF_edss_tfidf_logreg, 
        # LF_edss_tfidf_lda, LF_edss_tfidf_svc_rbf, LF_edss_tfidf_linear_svc

# These labelling functions can be run in current snorkel enviornment
def get_snorkel_compatible_lfs():
    return [LF_edss_19_int, LF_edss_19_dec, LF_edss_19_word, 
        LF_edss_19_roman]
