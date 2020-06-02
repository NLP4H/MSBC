"""
Snorkel labelling functions for sensory score
Note: This sensory subscore is not based on Neurostatus defns but 
based on heureustic information provided by Zhen based on advice given from MS clinicians
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import snorkel
from snorkel.labeling import labeling_function

def count_num_limbs(limb_list):

    limb_dict = {
        "hands":2,
        "right hand":1,
        "left hand":1,
        "ankles":2,
        "right ankle":1,
        "left ankle":1,
        "knees":2,
        "right knee":1,
        "left knee":1,
        "toes":2,
        "right toe":1,
        "left toe":1,
        "legs":2,
        "right leg":1,
        "left leg":1,
        "ribs":2,
        "rib ":1,
        "feet":2,
        "foot":1,
        "right foot":1,
        "left foot":1,
        "joints":2,
        "joint ":1,
        "elbows":2,
        "elbow":1,
        "right elbow":1,
        "left elbow":1,
        "shoulders":2,
        "shoulder":1,
        "arms":2,
        "wrists":2,
        "left wrist":1,
        "right wrist":1,
        "right arm":1,
        "left arm":1,
        "hips":2,
        "hip ":1,
        "upper extremity":1,
        "lower extremity":1,
        "upper extremities":2,
        "lower extremities":2,
        "limb ":1,
        "limbs":2,
        "right":1,
        "left":1,
        "right side":2,
        "left side":2,
        "entire right side":2,
        "entire left side":2,
        "4 limbs":4,
        "four limbs":4,
        "torso":2,
        "right finger":1,
        "left finger":1,
        "finger":2

    }
    count = 0
    for limb in limb_list:
        count += limb_dict[limb]
    return count

def get_sensory_info(note):
    
    limb_list = [
        r"hands",
        r"(?:right|left) hand",
        r"ankles",
        r"(?:right|left) ankle",
        r"knees",
        r"(?:right|left) knee",
        r"toes",
        r"(?:right|left) toe",
        r"legs",
        r"(?:right|left) leg",
        r"rib(?:\s|s)",
        r"feet",
        r"foot",
        r"(?:right|left) foot",
        r"joint(?:\s|s)",
        r"elbow",
        r"(?:right|left) elbow",
        r"shoulder"
        r"(?:right|left) arm",
        r"arms"
        r"hip(?:\s|s)",
        r"(?:upper|lower) extremit(?:y|ies)",
        r"limb(?:\s|s)",
        r"finger",
        r"(?:right|left) finger",
        r"wrists",
        r"(?:right|left) wrist",
        r"torso",
        r"right",
        r"left",
        r"(?:4|four) limbs",
        r"entire (?:right|left) side"
    ]
    
    p_limb = re.compile(r"|".join(limb_list), re.IGNORECASE)
    p = re.compile(r"sensation|vibrat(?:ion|ory)|temperature|coolness|(?:light)? touch|proprioception|position|pinprick|pin |pain ", re.IGNORECASE)
    p_level = re.compile(r"intact| normal|mild|minimal|moderate|marked|absent|significant|just detectable|no sensation|unable to feel", re.IGNORECASE)
    p_neg = re.compile(r"not |no ", re.IGNORECASE)
    p_adj = re.compile(r"loss of|lost(?: to)?|increased|abnormal|insensate|\d{2}\%|hyperaesthesia|decreased|decrease in|impaired|impairment|diminish(?:ed|ment)|reduced|reduction|blunting", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    type_list, level_list, adj_list, limb_list = [],[],[],[]
    for sent in sentences:
        
        if len(re.findall(p, sent)) > 0 and \
        len(re.findall(r"(F|f)acial", sent)) == 0:
            ##print(sent)
            type_list += re.findall(p, sent)
            adj_list += re.findall(p_adj, sent)
            level_list += re.findall(p_level, sent)
            limb_list += re.findall(p_limb, sent)
    
    type_list = [w.lower() for w in type_list]
    level_list = [w.lower() for w in level_list]
    adj_list = [w.lower() for w in adj_list]
    limb_list = [w.lower() for w in limb_list]
    
    ##print("Type:", list(set(type_list)))
    #print("Adj:", list(set(adj_list)))
    #print("Level: ", list(set(level_list)))
    #print("Limb: ", list(set(limb_list)))
    
    return list(set(type_list)), list(set(adj_list)), list(set(level_list)), list(set(limb_list))

def predict_vibration(note, limb_counts):
    score = -1
    p = re.compile(r"vibrat", re.IGNORECASE)
    p_mild = re.compile(r"mild|minimal", re.IGNORECASE)
    p_severe = re.compile(r"marked|absent|significant", re.IGNORECASE)
    p_neg = re.compile(r"normal|intact|unremarkable|reasonable", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(r"absent vibrat(?:ion|ory)", sent)) > 0:
            score = 3
            break
        
        if len(re.findall(p, sent)) > 0:
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 3
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                if limb_counts <= 2:
                    score = 1
                if limb_counts > 2:
                    score = 2
            # Moderate
            if len(re.findall(p_mild, sent)) == 0 and \
            len(re.findall(p_severe, sent)) == 0:
                if limb_counts <= 2:
                    score = 2
                if limb_counts > 2:
                    score = 3

    #print("Vibration:", score)
    return score

def predict_temperature(note, limb_counts):
    score = -1
    p = re.compile(r"temperature|coolness", re.IGNORECASE)
    p_mild = re.compile(r"mild|minimal", re.IGNORECASE)
    p_severe = re.compile(r"marked|absent|significant", re.IGNORECASE)
    p_neg = re.compile(r"normal|intact", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 3
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                if limb_counts <= 2:
                    score = 1
                if limb_counts > 2:
                    score = 2
            # Moderate
            if len(re.findall(p_mild, sent)) == 0 and \
            len(re.findall(p_severe, sent)) == 0:
                if limb_counts <= 2:
                    score = 2
                if limb_counts > 2:
                    score = 3
    
    
    
    #print("Temperature:", score)
    return score

def predict_proprioception(note, limb_counts):
    score = -1
    p = re.compile(r"proprioception", re.IGNORECASE)
    p_mild = re.compile(r"mild|minimal", re.IGNORECASE)
    p_severe = re.compile(r"marked|absent|significant", re.IGNORECASE)
    p_neg = re.compile(r"(?:normal|intact|unremarkable)(?: proprioception)?", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(r"reduced proprioception", sent)) > 0:
            score = 2
            break
        if len(re.findall(p, sent)) > 0:
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 4
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                if limb_counts <= 2:
                    score = 2
                if limb_counts > 2:
                    score = 3
            # Moderate
            if len(re.findall(p_mild, sent)) == 0 and \
            len(re.findall(p_severe, sent)) == 0:
                score = 3
    #print("Proprioception:", score)
    return score

def predict_touch(note, limb_counts):
    
    score = -1
    p = re.compile(r"(?:light)? touch", re.IGNORECASE)
    p_mild = re.compile(r"mild|minimal", re.IGNORECASE)
    p_severe = re.compile(r"marked|absent|significant", re.IGNORECASE)
    p_neg = re.compile(r"normal|intact", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 4
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                if limb_counts <= 2:
                    score = 2
                if limb_counts > 2:
                    score = 3
            # Moderate
            if len(re.findall(p_mild, sent)) == 0 and \
            len(re.findall(p_severe, sent)) == 0:
                if limb_counts <= 2:
                    score = 3
                if limb_counts > 2:
                    score = 4
    #print("Light touch:", score)
    return score

def predict_pinprick(note, limb_counts):
    score = -1
    p = re.compile(r"pinprick|pin ", re.IGNORECASE)
    
    p_mild = re.compile(r"mild|minimal|blunting", re.IGNORECASE)
    p_severe = re.compile(r"marked|absent|significant", re.IGNORECASE)
    p_neg = re.compile(r"normal |intact|unremarkable|did not report", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 3
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                if limb_counts <= 2:
                    score = 2
                if limb_counts > 2:
                    score = 3
            # Moderate
            if len(re.findall(p_mild, sent)) == 0 and \
            len(re.findall(p_severe, sent)) == 0:
                score = 3
    #print("Pinprick:", score)
    return score

def general_rules(note, limb_counts):
    score = -1
    limb_list = [
        "toe",
        "hand",
        "ankle",
        "knee",
        "toe",
        "leg",
        "rib",
        "feet|foot",
        "joint",
        "elbow",
        "hip",
        "extremit",
        "limb",
        "finger",
        "wrist",
    ]
    p_limb = re.compile(r"|".join(limb_list), re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        # Sensory Exam Normal
        if len(re.findall(r"Sensory (examination|exam)", sent)) > 0 and \
            len(re.findall(r"normal|intact", sent)) > 0 and \
            len(re.findall(p_limb, sent)) == 0:
            score = 0
            break
        
        if len(re.findall(r"insensate", sent)) > 0 and \
        len(re.findall(r"lower extremities", sent)) > 0:
            score = 5
            break
    
    #print("General rules:", score)
    return score

def select_neuro_exam(note):
    
    """
    Select only the Neurological Exam part of the note
    Input: a single piece of full note
    Output: neurological exam part of the note
    
    """

    p = re.compile(r"NEUROLOGICAL EXAMINATION:|EXAMINATION:|NEUROLOGICAL|(?:Neurological|neurological|neurologic|Neurologic) exam|On examination (day|today)")
    p1 = re.compile(r"Cranial|Visual|Vision")
    p2 = re.compile(r"examination|exam", re.IGNORECASE)
    p3 = re.compile(r"(?:EXAMINATION:|On examination today\,).*(?:IMPRESSION(?:\:| AND PLAN\:))")
    
    if len(re.findall(p3, note)) > 0:
        selected_note = re.findall(p3, note)[0]
        return selected_note

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

@labeling_function()  
# Prediction function
def LF_sensory_original(df_row):
    
    score = -1
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1
    note = df_row.text

    selected_note = select_neuro_exam(note)
    type_list, adj_list, level_list, limb_list = get_sensory_info(selected_note)
    limb_counts = count_num_limbs(limb_list)

    if edss_categorical == 0:
        #print("EDSS 0")
        score = 0
    
    
    elif level_list in [[' normal'], ['intact']] and \
    len(adj_list) == 0:
        #print("Sensory info normal")
        score = 0
    
        
    elif "no sensation" in level_list or \
    "unable to feel" in level_list:
        #print("No sensation")
        score = 4
    
    
    else:
        #print("Other situation")
        score = max(
            predict_pinprick(selected_note,limb_counts),
            predict_vibration(selected_note, limb_counts),
            predict_temperature(selected_note,limb_counts),
            predict_proprioception(selected_note,limb_counts),
            predict_pinprick(selected_note,limb_counts),
            general_rules(selected_note,limb_counts)
        )

    return score


def get_sensory_lfs():

    return [LF_sensory_original]