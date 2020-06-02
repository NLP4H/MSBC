"""
Snorkel labelling functions for pyramidal score
Note: This pyramidal subscore is not based on Neurostatus defns but 
based on heureustic information provided by Zhen based on advice given from MS clinicians
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

import snorkel
from snorkel.labeling import labeling_function

# predict muscle groups
# TODO: Other patterns, abbreviations
# re.compile(r"FDP|FDI|APB")
def predict_muscle_groups(note):
       
       # List of upper extremity muscle groups
       upper_lower_extremity = [
                   "finger extens", # Upper extremity
                   "finger abduct",
                   "finger grip",
                   "wrist flex", 
                   "wrist extens",
                   "hand abduct",
                   "elbow flex",
                   "elbow extens",
                   "arm abduct",
                   "digiti minimi",
                   "bicep",
                   "tricep",
                   "deltoid",
                   "interossei",  
                   "hip flex", # Lower extremity
                   "iliopsoa",
                   "knee extens",                   
                   "knee flex",
                   "quadricep",
                   "quads", 
                   "hamstring",
                   "gluteus minimus", 
                   "gluteus medius"
                   "leg abduct",
                   "abductor pollicis brevis", 
                   "extensor halluce",
                   "extensor digitorum",
                   "thigh flex", 
                   "foot extens", 
                   "ankle dorsiflex",
                   "foot dorsiflex",
                   "toe dorsiflex",
                   "dorsiflex",
                   "plantar flex"]
       
       # Pattern to capture possible muscle groups 
       p_muscle = re.compile("|".join(upper_lower_extremity), re.IGNORECASE) 
       p_grade = re.compile(r"(?:[0-5]\-)?[0-5](?:\+|\-)?(?:\/5)?(?:.|\s)") # Capture Grade 
       
       # Extract muslce groups aligned with grade
       sent_list = []
       sentences = sent_tokenize(note)
       for sent in sentences:
              if len(re.findall(p_muscle, sent)) > 0:
                     sent_list.append(sent)
       
       # If no muscle groups are extracted, assign Unknown
       if len(sent_list) == 0:
              score_pyramidal = -1
       
       # If muscle group is found
       elif len(sent_list) > 0:
              muscle_group = {}

              # If grade is found in the sentence
              for sent in sent_list:
                     # TODO: Take care of those "except for" cases

                     # See whether both muscle groups and grades are present
                     if len(re.findall(p_grade, sent)) > 0 and len(re.findall(p_muscle, sent)) > 0:
                            
                            # If only one muscle group and one grade is found, assign grade to the muscle group
                            if len(re.findall(p_grade, sent)) == 1 and len(re.findall(p_muscle, sent)) == 1 and len(re.findall(r"EDSS", sent)) == 0:
                                   muscle_group[re.findall(p_muscle, sent)[0]] = re.findall(p_grade, sent)[0]

                            # If more muscle groups are found
                            elif len(re.findall(p_muscle, sent)) > 1:
                                   # Split the sentence by "," and "and"
                                   phrases = re.split('and |, ', sent)
                                   # # print(phrases)
                                   for j in range(len(phrases)):

                                          # If both muscle group and grade is found in this phrase, align them
                                          if len(re.findall(p_grade, phrases[j])) > 0 and len(re.findall(p_muscle, phrases[j])) > 0:
                                                 muscle_group[re.findall(p_muscle, phrases[j])[0]] = re.findall(p_grade, phrases[j])[-1]

                                          # If muscle group is found grade is not found, refer to previous phrase
                                          if len(re.findall(p_grade, phrases[j])) == 0 and len(re.findall(p_muscle, phrases[j])) > 0:
                                                 if len(re.findall(p_grade, phrases[j-1])) > 0:
                                                        muscle_group[re.findall(p_muscle, phrases[j])[0]] = re.findall(p_grade, phrases[j-1])[0]
                                                        
                                                 elif len(re.findall(p_grade, phrases[j-2])) > 0:
                                                        muscle_group[re.findall(p_muscle, phrases[j])[0]] = re.findall(p_grade, phrases[j-2])[0]
              ## print(muscle_group)

              # If nothing is detected, assign Unknown
              if muscle_group == {}:
                     score_pyramidal = -1
              # If mus
              else:
                     score_pyramidal = calculate_pyramidal_score(muscle_group)

       return score_pyramidal


# Combine muscle groups and grades to calculate pyramidal score (based on strength)
def calculate_pyramidal_score(muscle_group):
       
       score_pyramidal = -1
       # Re-arange the values into grade 0-5
       grades = []
       for v in muscle_group.values():
              grades.append(int(v[0]))
       
       # Count no. of 0s 1s 2s 3s
       # Pyramidal 5 - 3+ group <= 2
       if grades.count(0) + grades.count(1) + grades.count(2) >= 3:
              score_pyramidal = 5

       # Pyramidal 4 - One group <= 2
       elif grades.count(0) + grades.count(1) + grades.count(2) >= 1:
              score_pyramidal = 4

       

       elif grades.count(0) == grades.count(1) == grades.count(2) == 0:

              # Pyramidal 3 - One group grade 3, 3 groups grade 4
              if grades.count(3) == 1 or grades.count(4) == 3:
                     score_pyramidal = 3

              # Pyramidal 2 - 2 groups grade 4
              if grades.count(4) == 2 and grades.count(3) == 0:
                     score_pyramidal = 2
       
       return score_pyramidal


# predict reflex status
def predict_reflex(note):
    
    score_reflex = -1
    p_reflex = re.compile(r"reflex", re.IGNORECASE)
    p_keyword = re.compile(r"brisk|sluggish|hyperreflexia", re.IGNORECASE) 

    # If keywords like "brisk" is found - reflex abnormal
    if len(re.findall(p_keyword, note)) > 0:
            score_reflex = 1

    # If reflex grade is not 2 (normal) - reflex abnormal
    elif len(re.findall(p_reflex, note)) > 0:
            sentences = sent_tokenize(note)
            for sent in sentences:
                    if len(re.findall(p_reflex, sent)) > 0:
                        if len(re.findall(r"[0-1]|[3-5]|zero|one", sent)) > 0:
                                score_reflex = 1
                                break
                        elif len(re.findall(r"diminish|downgoing|deep", sent)) > 0:
                                score_reflex = 1
                                break

    return score_reflex


# predict tone
def predict_tone(note):

       
        # mild/moderate/severe spasticity
       score_tone = -1
       p_tone = re.compile(r" spastic", re.IGNORECASE)
       p_level = re.compile(r"mild|subtle|minimal", re.IGNORECASE)
       
       sentences = sent_tokenize(note)
       for sent in sentences:
              # Eliminate cases of "bladder/no/not spasticity"
              if len(re.findall(p_tone, sent)) > 0 and len(re.findall(r"bladder| no |No |not|denies|without", sent)) == 0:
                    # "mild" - 2
                     if len(re.findall(p_level, sent)) > 0:
                            score_tone = 2
                            break
                    # "moderate" or "severe" - 3 
                     elif len(re.findall(p_level, sent)) == 0:
                            score_tone = 3
                            continue

       return score_tone


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
# Predict score pyramidal based on EDSS and rules
def LF_pyramidal_original(df_row):
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    note = df_row.text

    # By default UNKNOWN
    score = -1
    selected_note = select_neuro_exam(note)
    # edss_categorical based general rule
    # edss_categorical 0 - Pyramidal 0
    if edss_categorical== 0.0:
        score = 0
        # print("Score 0 (EDSS 0.0)")
        return score
    
    # EDSS 1 - Pyramidal 0 or 1
    if edss_categorical== 1: # 1.0
        
        # print("EDSS 1.0")
        # predict reflex
        score_reflex = predict_reflex(selected_note)
        # if score_reflex != -1: 
            # print("Reflex abnormalities detected")
        # predict tone, but maximum 1
        if predict_tone(selected_note) > 0:
            score_tone = max(1, predict_tone(selected_note))
        else:
            score_tone = predict_tone(selected_note)
        # if score_tone != -1:
            # print("Tone abnormalities detected")
        score = max(score_reflex, score_tone)
        # If no reflex or spascity, roughly predict whether there is mild weakness
        if score == -1:
             sentences = sent_tokenize(selected_note)
             for sent in sentences:
                    if len(re.findall(r"thumb|shoulder|hand|hip|leg", sent)) > 0 and len(re.findall(r"weakness|tingling|numbness", sent)) > 0:
                           # print("Strength abnormalities detected")
                           # print(sent)
                           score = 1
                           break
        
        # Set a default of 0
        if score == -1: score = 0
        return score
    
    # EDSS 1.5 - Pyramidal 0 or 1 or 2
    if edss_categorical== 2: # 1.5
        
        # print("EDSS 1.5")
        # Set a default of 0
        score = 0
        # predict reflex
        score_reflex = predict_reflex(selected_note)
        # if score_reflex != -1:
            # print("Reflex abnormalities detected")
        # predict tone, but maximum 2
        score_tone = min(2, predict_tone(selected_note))
        # if score_tone != -1:
            # print("Tone abnormalities detected")
        score = max(score_reflex, score_tone)
        # If no reflex or spascity, roughly predict whether mild weakness
        if score == -1:
             sentences = sent_tokenize(selected_note)
             for sent in sentences:
                    if len(re.findall(r"thumb|shoulder|hand|hip|leg", sent)) > 0 and len(re.findall(r"weakness|tingling|numbness", sent)) > 0:
                           # print("Strength abnormalities detected")
                           # print(sent)
                           score = 1
                           break
        
        # Set a default of 0
        if score == -1: score = 0
        return score


    if edss_categorical>= 13: # 7.0
        
        # print("EDSS greater than 7.0")
        # Set a default of 4 based on EDSS
        score = 4
        if len(re.findall(r"paraplegic|Paraplegic", selected_note)) > 0:
            # print("Found 'Paraplegic'")
            score = 5
            return score
        
        sentences = sent_tokenize(selected_note)
        for sent in sentences:
            # Lower limb 0 (score = 5 or 6)
            if len(re.findall(r"(?:no|minimal|not have any).*movement", sent)) > 0 and len(re.findall(r"legs|lower limb|lower extremi", sent)) > 0:
                # print("Found lower limb 0")
                score = 5
                # TODO: Find whether upper limb is also 0 and assign 6
                # if ...:
                # score = 6
                break
        return score
            


    # If general rule doesn't apply
    # TODO: Find tune this part
    else:
        # print("EDSS between 1.5 and 7.0")
        reflex_score = predict_reflex(selected_note)
        tone_score = predict_tone(selected_note)
        strength_score = predict_muscle_groups(selected_note)
        # print("Reflex:", reflex_score, "Tone: ", tone_score, "Strength: ", strength_score)
        
        # Prevent overprediction for EDSS = 2.0
        if edss_categorical== 3: # 2.0
            tone_score = min(2, tone_score)
            strength_score = min(2, strength_score)
        
        if edss_categorical== 5: # 3.0
            tone_score = min(3, tone_score)
            strength_score = min(3, strength_score)
        
        score = max(reflex_score, tone_score, strength_score)

    return score

def get_pyramidal_lfs():

    return [LF_pyramidal_original]