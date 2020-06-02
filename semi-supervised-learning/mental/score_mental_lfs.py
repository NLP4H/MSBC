"""
Snorkel labelling functions for mental subscore
NOTE: The rule-based LF is the original rulebased because the labels in the dataset 
follows heurestics outlined by MS clinicians, not the Neurostatus definitions.
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import snorkel
from snorkel.labeling import labeling_function

@labeling_function()    
def LF_mental_original(df_row):
    
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    note = df_row.text

    p = re.compile(r"subjective cognitive complaints", re.IGNORECASE)
    p2 = re.compile(r"Montreal Cognitive Assessment|MoCA", re.IGNORECASE)
    p3 = re.compile(r"(?:mild|Mild) cognitive challenge", re.IGNORECASE)
    p4 = re.compile(r"(?:mild|Mild) fatigue", re.IGNORECASE)
    p_neg = re.compile(r"No | no | deni|not have|not had", re.IGNORECASE)

    # Unknown by default
    score = -1
    
    if edss_categorical == 0:
        if len(re.findall(r"fatigue", note)) == 0:
            score = 0
        else:
            score = 1
    
    
    # MoCA test
    sentences = sent_tokenize(note)
    possible_sentences = []
    for sent in sentences:
        if len(re.findall(p2, sent)) > 0 and len(re.findall(r"30\/30", sent)) > 0:
            score = 0
            break
        if len(re.findall(p2, sent)) > 0 and len(re.findall(r"(?:25|26|27|28|29)\/30", sent)) > 0:
            score = 1
            break
        if len(re.findall(p2, sent)) > 0 and len(re.findall(r"(?:20|21|22|23|24)\/30", sent)) > 0:
            score = 2
            break
        if len(re.findall(p2, sent)) > 0 and len(re.findall(r"(?:10|11|12|13|14|15|16|17|18|19)\/30", sent)) > 0:
            score = 3
            break
            
        if len(re.findall(p3, sent)) > 0 or len(re.findall(p4, sent)) > 0:
            score = 1
            break
            
        if len(re.findall(p3, sent)) > 0 and len(re.findall(p4, sent)) > 0:
            score = 2
            break
        
        # Collect all sentences that have cognition|cognitive
        if len(re.findall(r"Cognition|cognition|Cognitive|cognitive", sent)) > 0:
            possible_sentences.append(sent)
            
    
    for sent in possible_sentences:
        if len(re.findall(p_neg, sent)) > 0 or len(re.findall(r"no longer has|significantly better", sent)) > 0:
            score = 0
            break
        elif len(re.findall(r"mild|Mild", sent)) > 0:
            score = 1
        elif len(re.findall(r"significant", sent)) > 0:
            score = 3
        else:
            score = 2
            
    # significant cognitive issue
    if len(re.findall(r"significant cognitive issue|progressive cognitive (?:and physical) decline", note)) > 0:
        if edss_categorical == 0:
            score = 1
        elif edss_categorical > 5: # 3.0
            score = 3
        else:
            score = 2

        
    if len(re.findall(p, note)) > 0:
        score = 1
        if edss_categorical == 3: # 2.0
           #df['score_brain_stem_subscore'][i] == 0 and \
           #df['score_cerebellar_subscore'][i] == 0 and \
           #df['score_ambulation_subscore'][i] == 0 and \
           #df['score_visual_subscore'][i] == 0 and \
           #df['score_pyramidal_subscore'][i] == 0 and \
           #df['score_sensory_subscore'][i] == 0 and \
           #df['score_bowel_bladder_subscore'][i] == 0:
            #print("Subjective cognitive complaints")
            score = 2
 
    return score

def get_mental_lfs():
    return [LF_mental_original]



# ---------------------------------OTHER ATTEMPTS BASED ON NEUROSTATUS DEFNS ---------------------------------
# Based on Neurostatus Definitions.pdf
def get_scores_array():
    scores= []
    score_0 = [0,0]
    score_1 = [1,1]
    score_2 = [2,2]
    score_3 = [3,2]
    score_4 = [4,2]
    score_5 = [5,2]
    scores = [score_0, score_1, score_2, score_3, score_4, score_5]
    
    return scores

def vote_scores(mentation, fatigue, mentation_weight, fatigue_weight):
    scores = get_scores_array()
    score_predictions = []
    for score in scores:
        predictions = []

        # check mobility_mode
        if mentation == score[0]:
            predictions.append(mentation_weight)
        else:
            predictions.append(0)
        
        # check edss 
        if fatigue == score[1]:
            predictions.append(fatigue_weight)
        else:
            predictions.append(0)
        
        score_predictions.append(predictions)
    
    return score_predictions

def predict_mentation(note, edss_categorical):
    """
    0: None 
    - moca : 30\/30
    1: signs only - not apparent, not significant
    - moca : (?:25|26|27|28|29)\/30
    2: mild - mild changes in mentation
    - impaired ability to follow rapid course of association or surveying complex matters
    - impaired judgement in demanding situation
    - capable of handling routine daily activities, but can't tolerate additional stressors
    - intermittently symptomatic even to normal levels of stress
    - reduced performance
    - negligence due to obliviousness or fatigue
    - moca: (?:20|21|22|23|24)\/30
    3: moderate
    - definite abnormalities on brief mental status
    - still oriented to person, place and time
    - moca: (?:10|11|12|13|14|15|16|17|18|19)\/30
    4: marked
    - not oriented in one or two spheres (person, place or time), marked effect on lifestyle
    5: dementia
    - confusion, complete disorientation
    """
    score = -1

    p_moca = re.compile(r"(?=)Montreal Cognitive Assessment|MoCA", re.IGNORECASE)
    p_moca_none =  re.compile(r"30\/30")
    p_moca_mild =  re.compile(r"(?:20|21|22|23|24)\/30")
    p_moca_moderate = re.compile(r"(?:10|11|12|13|14|15|16|17|18|19)\/30")

    # if mention of MOCA test
    if len(re.findall(p_moca, note)) > 0:
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_moca, sent)
            if mentions != None:
                # if none
                if len(re.findall(p_moca_none, sent)) > 0:
                    score = 0
                    break
                
                # if mild
                elif len(re.findall(p_moca_mild, sent)) > 0:
                    score = 1
                    break
                
                # if moderate
                elif len(re.findall(p_moca_moderate, sent)) > 0:
                    score = 2
                    break
    
    p_mentation = re.compile(r"(?=)mentation|cognitive", re.IGNORECASE)
    p_none = re.compile(r"(?=)no longer has|significantly better", re.IGNORECASE)
    p_signs_only = re.compile(r"(?=)signs of|not apparent|not significant|subjective cognitive complaints", re.IGNORECASE)
    p_mild = re.compile(r"(?=)mild|slight|impaired judgement|impaired ability|intermittently symptomatic|negligence", re.IGNORECASE)
    p_moderate = re.compile(r"(?=)moderate|definite abnormalities", re.IGNORECASE)
    p_marked = re.compile(r"(?=)severe|serious|extensive|significant|progressive", re.IGNORECASE)
    p_dementia = re.compile(r"(?=)dementia", re.IGNORECASE)

    # if moca score not found and there is mention of mentation
    if score == -1 and len(re.findall(p_mentation, note)) > 0:
        # get p_mentation level
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_mentation, sent)
            
            if mentions != None:

                 # if dementia
                if len(re.findall(p_dementia, sent)) > 0:
                    score = 5
                    break
                
                # if mild
                elif len(re.findall(p_mild, sent)) > 0:
                    score = 2
                    break
                
                # if none
                elif len(re.findall(p_none, sent)) > 0:
                    score = 0
                    break
                    
                # if marked
                elif len(re.findall(p_marked, sent)) > 0:
                    score = 4
                    break
                    
                # if signs
                elif len(re.findall(p_signs_only, sent)) > 0:
                    score = 1
                    break
                    
                # if moderate
                elif len(re.findall(p_moderate, sent)) > 0:
                    score = 3
                    break
                
                # else, guess mild
                else:
                    score = 1

    p_mentation2 = re.compile(r"(?=)cognitive complain|cognitive issues|cognitive impairment", re.IGNORECASE)
    
    if score == -1:
        if len(re.findall(p_mentation2, note)) > 0:
            if edss_categorical == 3:
                score = 2
            else:
                score = 1

    p_significant_cognitive = re.compile(r"significant cognitive issue|progressive cognitive (?:and physical) decline", re.IGNORECASE)
    
    # significant cognitive issue
    if len(re.findall (p_significant_cognitive, note)) > 0:
        if edss_categorical == 0:
            score = 1
        elif edss_categorical > 5:
            score = 3
        else:
            score = 2

    return score

def predict_fatigue(note):
    """
    0: none
    1: mild - doesn't interfere with daily activities
    2 / 3: moderate / severe - interferes, could limit daily activities ( < 50% reduction)
    """

    score = -1

    p_fatigue = re.compile(r"(?=)fagitue|tired", re.IGNORECASE)
    p_none =  re.compile(r"no fatigue")
    p_mild =  re.compile(r"mild|a little|a bit|slight")
    
    # if mention of fatigue test
    if len(re.findall(p_fatigue, note)) > 0:
        sentences = sent_tokenize(note)

        for sent in sentences:
            mentions = re.search(p_fatigue, sent)

            if mentions != None:
                # if none
                if len(re.findall(p_none, note)) > 0:
                    score = 0
                    break
                
                # if mild
                elif len(re.findall(p_mild, note)) > 0:
                    score = 1
                    break

                else:
                    score =2
                    break
    return score

@labeling_function()  
def LF_mental_average(df_row):
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    note = df_row.text

    mentation = predict_mentation(note, edss_categorical)
    fatigue = predict_fatigue(note)
    score = -1

    # get matrix of possible scores given mentation and fatigue
        # mentatation has higher weight, more significant indicator
    scores = vote_scores(mentation, fatigue, 2, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mentation and fatigue into at least 1
    if max_vote > 0:
        # get all scores that have max score
        max_scores = np.asarray(np.where(vote_per_score == max_vote))[0]

        # if only 1 score has max score, then 
        if len(max_scores) == 1:
            score = max_scores[0]
            
        else:
            # get score that is closest to the average score
            avg_score = np.average(max_scores)
            score = max_scores[np.abs(max_scores - avg_score).argmin()]
        
    return score

def predict_mentation(note, edss_categorical):
    """
    0: None 
    - moca : 30\/30
    1: signs only - not apparent, not significant
    - moca : (?:25|26|27|28|29)\/30
    2: mild - mild changes in mentation
    - impaired ability to follow rapid course of association or surveying complex matters
    - impaired judgement in demanding situation
    - capable of handling routine daily activities, but can't tolerate additional stressors
    - intermittently symptomatic even to normal levels of stress
    - reduced performance
    - negligence due to obliviousness or fatigue
    - moca: (?:20|21|22|23|24)\/30
    3: moderate
    - definite abnormalities on brief mental status
    - still oriented to person, place and time
    - moca: (?:10|11|12|13|14|15|16|17|18|19)\/30
    4: marked
    - not oriented in one or two spheres (person, place or time), marked effect on lifestyle
    5: dementia
    - confusion, complete disorientation
    """
    score = -1

    p_moca = re.compile(r"(?=)Montreal Cognitive Assessment|MoCA", re.IGNORECASE)
    p_moca_none =  re.compile(r"30\/30")
    p_moca_mild =  re.compile(r"(?:20|21|22|23|24)\/30")
    p_moca_moderate = re.compile(r"(?:10|11|12|13|14|15|16|17|18|19)\/30")

    # if mention of MOCA test
    if len(re.findall(p_moca, note)) > 0:
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_moca, sent)
            if mentions != None:
                # if none
                if len(re.findall(p_moca_none, sent)) > 0:
                    score = 0
                    break
                
                # if mild
                elif len(re.findall(p_moca_mild, sent)) > 0:
                    score = 1
                    break
                
                # if moderate
                elif len(re.findall(p_moca_moderate, sent)) > 0:
                    score = 2
                    break
    
    p_mentation = re.compile(r"(?=)mentation|cognitive", re.IGNORECASE)
    p_none = re.compile(r"(?=)no longer has|significantly better", re.IGNORECASE)
    p_signs_only = re.compile(r"(?=)signs of|not apparent|not significant|subjective cognitive complaints", re.IGNORECASE)
    p_mild = re.compile(r"(?=)mild|slight|impaired judgement|impaired ability|intermittently symptomatic|negligence", re.IGNORECASE)
    p_moderate = re.compile(r"(?=)moderate|definite abnormalities", re.IGNORECASE)
    p_marked = re.compile(r"(?=)severe|serious|extensive|significant|progressive", re.IGNORECASE)
    p_dementia = re.compile(r"(?=)dementia", re.IGNORECASE)

    # if moca score not found and there is mention of mentation
    if score == -1 and len(re.findall(p_mentation, note)) > 0:
        # get p_mentation level
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_mentation, sent)
            
            if mentions != None:

                 # if dementia
                if len(re.findall(p_dementia, sent)) > 0:
                    score = 5
                    break
                
                # if mild
                elif len(re.findall(p_mild, sent)) > 0:
                    score = 2
                    break
                
                # if none
                elif len(re.findall(p_none, sent)) > 0:
                    score = 0
                    break
                    
                # if marked
                elif len(re.findall(p_marked, sent)) > 0:
                    score = 4
                    break
                    
                # if signs
                elif len(re.findall(p_signs_only, sent)) > 0:
                    score = 1
                    break
                    
                # if moderate
                elif len(re.findall(p_moderate, sent)) > 0:
                    score = 3
                    break
                
                # else, guess mild
                else:
                    score = 1

    return score

def predict_fatigue(note):
    """
    0: none
    1: mild - doesn't interfere with daily activities
    2 / 3: moderate / severe - interferes, could limit daily activities ( < 50% reduction)
    """

    score = -1

    p_fatigue = re.compile(r"(?=)fagitue|tired", re.IGNORECASE)
    p_none =  re.compile(r"no fatigue")
    p_mild =  re.compile(r"mild|a little|a bit|slight")
    
    # if mention of fatigue test
    if len(re.findall(p_fatigue, note)) > 0:
        sentences = sent_tokenize(note)

        for sent in sentences:
            mentions = re.search(p_fatigue, sent)

            if mentions != None:
                # if none
                if len(re.findall(p_none, note)) > 0:
                    score = 0
                    break
                
                # if mild
                elif len(re.findall(p_mild, note)) > 0:
                    score = 1
                    break

                else:
                    score =2
                    break
    return score


@labeling_function()  
def LF_mental_average(df_row):
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    note = df_row.text
    mentation = predict_mentation(note, edss_categorical)
    fatigue = predict_fatigue(note)
    
    # get matrix of possible scores given mentation and fatigue
        # mentatation has higher weight, more significant indicator
    scores = vote_scores(mentation, fatigue, 2, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mentation and fatigue into at least 1
    if max_vote > 0:
        # get all scores that have max score
        max_scores = np.asarray(np.where(vote_per_score == max_vote))[0]

        # if only 1 score has max score, then 
        if len(max_scores) == 1:
            score = max_scores[0]
            
        else:
            # get score that is closest to the average score
            avg_score = np.average(max_scores)
            score = max_scores[np.abs(max_scores - avg_score).argmin()]

    return score

"""
RULES
According to neurostatus defns, mental (cerebral) functions depends on 

1. Decrease in mentation:
0: None 
1: signs only - not apparent, not significant
2: mild - mild changes in mentation
- impaired ability to follow rapid course of association or surveying complex matters
- impaired judgement in demanding situation
- capable of handling routine daily activities, but can't tolerate additional stressors
- intermittently symptomatic even to normal levels of stress
- reduced performance
- negligence due to obliviousness or fatigue
3: moderate
- definite abnormalities on brief mental status
- still oriented to person, place and time
4: marked
- not oriented in one or two spheres (person, place or time), marked effect on lifestyle
5: dementia
- confusion, complete disorientation

2. Fatigue
0: none
1: mild - doesn't interfere with daily activities
2: moderate - interferes, but doesn't limit daily activities ( < 50% reduction)
3: severe: - significant limitations in daily activities (> 50% reduction)

SCORING:
0: 
- mentation = 0
- fatigue = 0

1:
- mentation = 1 
- fatigue = 1

2: 
- mentation = 2
- fatigue = 2 or 3

3: 
- mentation = 3
- fatigue = 2 or 3

4: 
- mentation = 4
- fatigue = 2 or 3

5: 
- mentation = 5
- fatigue = 2 or 3


"""