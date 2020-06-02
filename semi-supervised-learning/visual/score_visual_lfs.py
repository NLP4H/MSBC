"""
Snorkel labelling functions for visual score
Note: This visual subscore is not based on Neurostatus defns but 
based on heureustic information provided by Zhen based on advice given from MS clinicians
"""
import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

import snorkel
from snorkel.labeling import labeling_function

def predict_visual_acuity(note):

    """ 
    Rules
    1: 20/20 – 20/30 
    2: 20/30 – 20/60 
    3: 20/60 – 20/100 
    4: 20/100 – 20/200 
    
    Input: note
    Returns: raw visual acuity score
    """
    score = -1
    # Pattern
    p = re.compile(r" 20\/\d{2,3}")
    p2 = re.compile(r"visual acuity", re.IGNORECASE)
     
    if len(re.findall(p, note)) > 0:
        # List of possible visual acuities in each note
        visual_acuities = []
        for acuity in re.findall(p, note):
            visual_acuities.append(int(acuity[4:]))

        # Take the worst disability
        worst_eye =  max(visual_acuities)
        best_eye = min(visual_acuities)
        
        # vision improvement -> remove the worst one
        sentences = sent_tokenize(note)
        for sent in sentences:
            # in each sentence, look for visual aquity number and "vision improved" token
            if len(visual_acuities) > 1 and len(re.findall(r"(?:Vision|vision)", sent)) > 0 and len(re.findall(r"improv", sent)) > 0:
                # If originally is finger counting, than no use to remove
                if len(re.findall(r"finger counting vision", note)) > 0:
                    break  
                else:
                    visual_acuities.remove(max(visual_acuities))
                    worst_eye =  max(visual_acuities)
                    break
            if len(visual_acuities) > 1 and len(re.findall(p, sent)) > 0 and len(re.findall(r"improv", sent)) > 0:
                visual_acuities.remove(max(visual_acuities))
                worst_eye =  max(visual_acuities)
                break
            
            # Vision recover
            if len(re.findall(r"(?:Vision|vision) recover", sent)) > 0:
                if len(re.findall(p, sent)) > 0:
                    visual_acuities = []
                    for acuity in re.findall(p, sent):
                        visual_acuities.append(int(acuity[4:]))
                    worst_eye =  max(visual_acuities)
                    best_eye = min(visual_acuities)
                    break
                else:
                    score = 0
                    # print("Visual Acuity: ", score)
                    return score

        # print("worst:", worst_eye)
        # print("best:", best_eye)
        # 20/20 normal
        if worst_eye == 20:
            score = 0
            # print("Visual Acuity: ", score)
            return score
        

        # 1: 20/20 – 20/30 
        elif worst_eye > 20 and worst_eye <= 30:
            score = 1
            # print("Visual Acuity: ", score)
            return score
        # 2: 20/30 – 20/60  
        elif worst_eye > 30 and worst_eye <= 60:
            score = 2
            # print("Visual Acuity: ", score)
            return score
        # 3: 20/60 – 20/100 
        elif worst_eye > 60 and worst_eye <= 100:
            score = 3
            # print("Visual Acuity: ", score)
            return score
        # 4: 20/100 – 20/200 
        elif (worst_eye > 100 and worst_eye <= 200) or \
            (worst_eye != best_eye and worst_eye > 60 and worst_eye <= 100 and best_eye > 60 and best_eye <= 100):
            score = 4
            # print("Visual Acuity: ", score)
            return score
        # 5: > 200
        elif (worst_eye > 200) or \
            (worst_eye != best_eye and worst_eye > 100 and worst_eye <= 200 and best_eye > 60 and best_eye <= 200):
            score = 5
            # print("Visual Acuity: ", score)
            return score
            
        # 6: worst eye > 200, best eye >= 60
        elif (worst_eye > 200):
            score = 6
            # print("Visual Acuity: ", score)
            return score
        
        
    # "Visual acuity" is detected
    elif len(re.findall(p2, note)) > 0:
        sentences = sent_tokenize(note)
        for sent in sentences:
            if len(re.findall(p2, sent)) > 0 and len(re.findall(r"normal|Normal", sent)) > 0:
                score = 0
                # print("Visual Acuity: ", score)
                return score
    
    # print("Visual Acuity: ", score)
    return score

def predict_pallor(note):
    
    """
    Check whether there's disc pallor
    Input: note
    Returns: score for disc pallor (maximum 1)
    
    """

    # Patterns
    p = re.compile(r" disk | disc |fundoscopy| fundi | fundus|optic nerve", re.IGNORECASE)
    p_neg = re.compile(r" no | not |normal|unremarkable|crisp", re.IGNORECASE)
    p_abnormal = re.compile(r"pallor|pale", re.IGNORECASE)
    
    # Predictions
    score = -1
    sentences = sent_tokenize(note)
    for sent in sentences:

        if len(re.findall(r"optic atrophy", sent)) > 0:
            score = 1
            break
        
        if len(re.findall(r"temporal pallor|significant pallor|bilateral optic disc pallor", sent)) > 0:
            score = 1
            break
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            elif len(re.findall(p_abnormal, sent)) > 0:
                score = 1
                break
    
    # print("Pallor:", score)
    return score

def predict_scotoma(note):
    
    """
    Check scotoma
    0: normal
    1: small / no mention of size
    2: large
    Input: note
    Returns: score for scotoma
    
    """
    
    # Patterns
    p = re.compile(r"scotoma", re.IGNORECASE)
    p_neg = re.compile(r" no | deni|not have|not had", re.IGNORECASE)

    # Predictions
    score = -1
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            # print(sent)

            # Negation
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            # Large
            elif len(re.findall(r"large|Large", sent)) > 0:
                score = 2
                break
            else:
                score = 1
                break
    
    # print("Scotoma: ", score)
    return score

def predict_visual_fields(note):
    """
    Outputs: 
    0: if no change in visual field
    1: if visual field got worst
    """
    p = re.compile(r"visual field", re.IGNORECASE)
    p_neg = re.compile(r"full|intact|normal")
    # p2 = re.compile(r"hemianopsia", re.IGNORECASE)
    
    score = -1
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
            elif len(re.findall(r"restrict", sent)) > 0:
                score = 1

    # print("Visual Fields: ", score)
    return score

def general_rule(note):

    """
    Zhen's heurestics (developed through meetings with MS clinicians who label)
    
    Apply general rules where there's no specific description in the notes
    1. Finger Counting
    2. Light Perception
    
    """
    
    # Normal
    # Some level of blindness
    
    # finger counting

    p1 = re.compile(r"count finger acuity|remains blind|left blind|right blind", re.IGNORECASE)
    score = -1
    sentences = sent_tokenize(note)
    # TODO: Black and white|shapes and shadows
    for sent in sentences:
        
        # Normal
        if len(re.findall(r"no visual symptom", sent)) > 0:
            # print("No visual symptons")
            score = 0
            break
        if len(re.findall(r"neurological exam", sent)) > 0 and len(re.findall(r"normal", sent)) > 0:
            # print("Neurological exam normal")
            score = 0
            break
        if len(re.findall(r"otherwise|Otherwise", sent)) > 0 and len(re.findall(r"normal", sent)) > 0 and len(re.findall(r"visual|vision", sent)) == 0:
            score = 0
            break
            
        if len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"based on sensory", sent)) > 0:
            score = 0
            break
        
        # Abnormal
        if len(re.findall(p1, sent)) > 0:
            # print("Blind/Finger counting")
            score = 6
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"foot", sent)) > 0:
            # print("Finger counting 1 ft")
            score = 5
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"2 feet|two feet|3 feet|three feet", sent)) > 0:
            # print("Finger counting 2/3 ft")
            score = 4
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"light perception", sent)) > 0:
            # print("Finger counting & light perception")
            score = 6
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s4", sent)) > 0 and len(re.findall(r"vision alone", sent)) > 0:
            # print("EDSS 4 related to vision")
            score = 6
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s3", sent)) > 0 and len(re.findall(r"vision|visual sign", sent)) > 0:
            # print("EDSS 3 related to vision")
            score = 4
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s2", sent)) > 0 and len(re.findall(r"vision|visual sign", sent)) > 0:
            score = 2
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s4", sent)) > 0 and len(re.findall(r"loss of vision", sent)) > 0:
            # print("EDSS 4 related to vision")
            score = 4
            break
        phrases = sent.split(",")
        for phrase in phrases:
            if len(re.findall(r"vision|visual", phrase)) > 0 and len(re.findall(r"significant", phrase)) > 0 and len(re.findall(r"impair", phrase)) > 0:
                if len(re.findall(r"improv", note)) > 0:
                    break

                score = 6
                break
    # print("General Rule: ", score)
    return score

def select_neuro_exam(note):
    """
    Function used for Zhen's heurestics
    """
    p = re.compile(r"NEUROLOGICAL EXAMINATION:|EXAMINATION:|NEUROLOGICAL|(?:Neurological|neurological|neurologic|Neurologic) examination")
    p1 = re.compile(r"Cranial|Visual|Vision|On examination day")
    p2 = re.compile(r"examination|exam", re.IGNORECASE)

    sentences = sent_tokenize(note)
    start_index = 0
    
    if len(re.findall(p, note)) > 0:
        for j in range(len(sentences)):
            if len(re.findall(p, sentences[j])) > 0:
                # start index = first sentence to mention neurological exam
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
def LF_visual_original(df_row):
    """
    Visual subscore prediction based on Zhen's heurestics (developed through meeting with MS clinicians)
    Visual subscore is determined from the highest potential visual subscore from general_rule, or outputs from predict_visual_acuity, predict_pallor, predict_scotoma and predict_visual_fields
    This doesn't match with the neurostatus definitions, but seems to be a heurestics that's applied when labelling the function
    This apparently gives higher accuracy than just following neurostatus defns
    """

    note = df_row.text
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # Unknown by default
    score = -1
    selected_note = select_neuro_exam(note)

    # EDSS = 0 all scores 0
    if edss_categorical == 0:
        score = 0

    else:
        score = max(general_rule(selected_note), predict_visual_acuity(selected_note), predict_pallor(selected_note), predict_scotoma(selected_note), predict_visual_fields(selected_note))

    return score

def get_visual_lfs():
    # Uncomment to test just new LFs

    return [LF_visual_original]
# RULES
# visual subscore depends on visual acuity, visual fields, scotoma, disc pallor

# Visual acuity
# 1: 20/20 – 20/30 
# 2: 20/30 – 20/60 
# 3: 20/60 – 20/100 
# 4: 20/100 – 20/200

# disk pallor:
# 0: none
# 1: present

# scotom,
# 0: normal
# 1: small / no mention of size
# 2: large

# visual fields
# 0: healthy
# 1: decline / restricted

# score 0
    # disc pallor: N/A = 0
    # scotoma: N/A = 0
    # visual field: N/A
    # visual acuity: normal = 1

# score 1
    # if either one or all (and/or)
    # disc pallor: true = 1
    # scotoma: small = 1
    # visual field: N/A = 0
    # visual acuity: 20/30 (0.67) - 20/20 (1.0) of worse eye = 2

# score 2:
    # disc pallor: N/A = 0
    # scotoma: N/A = 0
    # visual field: N/A = 0
    # visual acuity: 20/30(0.67) - 20/59(0.34) of worse eye with maximal visual acuity (corrected) = 2

# score 3:
    # disc pallor: N/A = 0
    # scotoma: large = 2
    # visual field: moderate decrease = 1
    # visual acuity: 20/60(0.33) - 20/99(0.21) of worse eye with maximal visual acuity (corrected) = 3

# score 4:
    # disc pallor: N/A = 0
    # scotoma: N/A
    # visual field: decrease in worse eye
    # visual acuity: 
        # 20/100(0.2) - 20/200(0.1) of worse eye with maximal visual acuity (corrected) = 4
        # grade 3
        # < 20/60 (0.33) for better eye = 1, 2

# score 5:
    # disc pallor: N/A = 0
    # scotoma: N/A = 0
    # visual field: N/A = 0
    # visual acuity: 
        # < 20/200 (0.1) for worse eye with maximal visual acuity = 1,2,3,4
        # grade 4
        # < 20/60 (0.33) for better eye with maximal visual acuity = 1,2

# score 6:
    # disc pallor: N/A = 0
    # scotoma: N/A = 0
    # visual field: N/A =0
    # visual acuity: 
        # grade 5
        # < 20/60 (0.33) for better eye with maximal visual acuity = 1,2