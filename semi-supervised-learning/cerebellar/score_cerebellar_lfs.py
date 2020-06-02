
"""
Snorkel labelling functions for cerebellar score
Note: This cerebellar subscore is not based on Neurostatus defns but 
based on heureustic information provided by Zhen based on advice given from MS clinicians
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import snorkel
from snorkel.labeling import labeling_function

def predict_heel_to_shin(note):
    #print("Heel-to-shin")

    p = re.compile(r"heel\-shin|heel\-to\-shin|heel to shin", re.IGNORECASE)
    p_mild = re.compile(r"mild|minimal|subtle|slight|minor", re.IGNORECASE)
    p_severe = re.compile(r"significant|severe|worsening|prominent|marked|worst", re.IGNORECASE)
    p_neg = re.compile(r" not | no | deni|without|unremarkable|normal|well performed|performed well|good|intact",  re.IGNORECASE)

    score = -1
    sentences = sent_tokenize(note)

    for sent in sentences:
            
        if len(re.findall(p, sent)) > 0 and (len(re.findall(p_neg, sent)) == 0):
            # moderate/severe 3
            if len(re.findall(p_severe, sent)) > 0 or len(re.findall(r"unable to perform", sent)) > 0:
                score = 3
                #print("severe")
                break
            # mild/minimal 2
            elif len(re.findall(p_mild, sent)) > 0:
                score = 2
                #print("mild")
                break             
            # reasonable 1
            elif len(re.findall(r"reasonabl", sent)) > 0:
                score = 1
                #print("reasonable")
                break
            elif len(re.findall(r"assess", sent)) > 0:
                score = 2
                #print("cannot be assessed")
                break
            elif len(re.findall(r"moderate|Moderate|impair|Impair|difficult", sent)) > 0:
                score = 3
                #print("moderate")
                break
    #print(score)
    return score

def predict_ataxia(note):
    #print("ataxia/dysmetria")
    p = re.compile(r"ataxia|ataxic|dysmetria", re.IGNORECASE)
    p2 = re.compile(r"(?:Gait|gait).*ataxic|some ataxia")
    # TODO: subtle/minor/minimal -> assign a 1 instead of 2
    p_mild = re.compile(r"(?:mild|minimal|subtle|slight|minor)( sensory| upper limb| lower limb| limb| gait| finger-nose)? (ataxia|ataxic|dysmetria)", re.IGNORECASE)    
    p_severe = re.compile(r"(?:significant|significantly|marked|markedly|bilateral|quite)( spastic and| upper limb| lower limb| limb| gait| finger-nose)? (ataxia|ataxic|dysmetria)")
    p_neg = re.compile(r"( possibly| no|not any evidence of|no evidence of)(?: upper limb| lower limb| limb| gait| finger-nose| appendicular)? (ataxia|ataxic)")
    p_neg2 = re.compile(r"not | no | deni|without|unremarkable|normal|well performed|performed well|good|intact", re.IGNORECASE)
    score = -1
    sentences = sent_tokenize(note)
   
    for sent in sentences:
        
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(r"No ataxia", sent)) > 0:
                score = 0
                break
            if len(re.findall(r"moderate|bilateral|heel-to-shin", sent)) > 0 and len(re.findall(r";|,", sent)) == 0:
                score = 3
            if len(re.findall(r"(?:marked|moderate)(?: limb|gait|finger-nose)? (?:ataxia|dysmetria)|ataxic gait|ataxic-paraparetic gait|moderately ataxic and paraparetic", sent)) > 0:
                score = 3
                break
            if len(re.findall(r"(?:mild|minimal|mildly|no clear)(?: left-sided(?: finger-to-nose)?| right-sided| left upper limb| heel-to-shin| bilateral| gait)? (?:ataxic|ataxia|limb ataxia|dysmetria)", sent)) > 0 or \
            len(re.findall(r"limb ataxia is minimal|mild gait ataxia", sent)) > 0:
                score = 1
                break

            if len(re.findall(p_neg2, sent)) == 0 and len(re.findall(r"limb", sent)) > 0:
                score = 3
                break
            if len(re.findall(p_neg2, sent)) > 0:
                # if negation is after the detected pattern, it is not related
                if sent.index(re.findall(p_neg2, sent)[0]) > sent.index(re.findall(p, sent)[0]):
                    score = 3
                else:
                    score = 0
            if len(re.findall(r"mild|minimal|subtle|slight|minor", sent)) > 0 and len(re.findall(r";|,", sent)) == 0:
                score = 2
            
            if len(re.findall(r"EDSS", sent)) > 0:
                if len(re.findall(r"mild",sent)) == 0:
                    score = 1
                else:
                    score = 3
                break
            
        if len(re.findall(p_neg, sent)) > 0:
            score = 0
        if len(re.findall(r"finger-to-nose imprecision|impaired finger-to-nose bilaterally", sent)) > 0:
            score = 2
            break
        if len(re.findall(p_severe, sent)) > 0:
            if len(re.findall(r"not convinced|did not (?:show|reveal)", sent)) > 0:
                score = 0
                break
            
            score = 4
            #print("severe")
            break
        if len(re.findall(p_mild, sent)) > 0:
            #print("mild")
            score = 2
            break
        if len(re.findall(p2, sent)) > 0:
            score = 2
            break
        


    #print(score)
    return score

def predict_tremor(note):
    #print("tremor")
    score = -1
    p = re.compile(r"intention tremor|postural tremor", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p, sent)) > 0:
                score = 1
                #print("1")
                break
    #print(score)
    return score

def predict_tandem_gait(note):
    #print("tandem gait")
    # Normal -> 0
    # Poor/instability -> 1
    # Can't do it at all -> 2
    p_mild = re.compile(r"mild|minimal|subtle|slight|minor", re.IGNORECASE)
    p_severe = re.compile(r"significant|severe|worsening|prominent|marked|worst", re.IGNORECASE)
    p_neg = re.compile(r" not |no | deni|within normal limits|without|unremarkable|normal|well\-performed|well performed|performed well|good|intact",  re.IGNORECASE)

    # patterns
    p = re.compile(r"tandem gait|tandem", re.IGNORECASE)
    
    score = -1
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p_mild, sent)) > 0:
                score = 1
                #print("mild")
                break
            elif len(re.findall(r"not able to perform|unable to perform|unable to do|cannot perform|could not perform|could not be performed", sent)) > 0:
                score = 3
                #print("can't perform")
                break
            elif len(re.findall(p_neg, sent)) > 0:
                score = 0
                #print("negation 0")
                break
            elif len(re.findall(r"impair|difficult", sent)) > 0:
                score = 2
                #print("moderate/impaired")
                break
    #print(score)
    return score

def select_neuro_exam(note):
    

    p = re.compile(r"NEUROLOGICAL EXAMINATION:|EXAMINATION:|NEUROLOGICAL|(?:Neurological|neurological|neurologic|Neurologic) examination")
    p1 = re.compile(r"Cranial|Visual|Vision|On examination day")
    p2 = re.compile(r"examination|exam", re.IGNORECASE)

    sentences = sent_tokenize(note)
    start_index = 0
    
    if len(re.findall(p, note)) > 0:
        for j in range(len(sentences)):
            if len(re.findall(p, sentences[j])) > 0:
                start_index = j
    else:
    
        for j in range(len(sentences)):
            if len(re.findall(p1, sentences[j])) > 0:
                start_index = j
                break
            elif len(re.findall(p2, sentences[j])) > 0:
                start_index = j
                break

    selected_note = " ".join([sentences[j] for j in range(start_index, len(sentences))])

    return selected_note

def general_rule(note):
    
    
    #print("General Rule: ")
    score = -1
    p = re.compile(r"(?:neurological|neurologic)? exam", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0 and len(re.findall(r"normal", sent)) > 0:
            score = 0
            break
    #print(score)
    return score

# Calculate cerebellar based on Zhen's original rule-based code
@labeling_function()     
def LF_cerebellar_original(df_row):
    note = df_row.text
    
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # EDSS = 0
    if edss_categorical == 0.0:       
        score = 0
        return score
    
    # EDSS != 0
    else:
        selected_note = select_neuro_exam(note)
        score = max(general_rule(selected_note), predict_ataxia(selected_note), predict_tremor(selected_note), predict_heel_to_shin(selected_note), predict_tandem_gait(selected_note))
        if score == 4 and edss_categorical < 7: # 4.0
            score = 3
        if score == 4 and edss_categorical <= 3.0: # 5.0
            score = 2        
        return score

def get_cerebellar_lfs():
    return [LF_cerebellar_original]

