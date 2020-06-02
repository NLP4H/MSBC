"""
Snorkel labelling functions for bowel bladder subscore
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import snorkel
from snorkel.labeling import labeling_function

def get_scores_array():
    """
    Get matrix of possibles ranges of urinary hesitancy and retention, urinary urgency and incontinence, bladder catheterisation, bowel dysfunction for each ambulation score
    - Based on Neurostatus Definitions.pdf
    Output: matrix of possible ranges of urinary hesitancy and retention, urinary urgency and incontinence, bladder catheterisation, bowel dysfunction per score
    """

    scores= []
    score_0 = [0, 0, 0, 0]
    score_1 = [1,1,0,0]
    score_2 = [2,2,0,0]
    score_3 = [4,3,1,3]
    score_4 = [4,3,2,3]
    score_5 = [4,4,3,4]
    scores = [score_0, score_1, score_2, score_3, score_4, score_5]
    
    return scores

def vote_scores(uri_hesitancy, uri_urgency, bladder_cath, bowel_dys, uri_hesitancy_weight, uri_urgency_weight, bladder_cath_weight, bowel_dys_weight):
    """
    Given urinary hesitancy and retention, urinary urgency and incontinence, bladder catheterisation, bowel dysfunction  prediction of a note --> vote on which scores the note could be
    Inputs:
        uri_hesitancy = predicted urinary hesitancy
        uri_urgency = predicted urinary urgency
        bladder_cath = predicted bladder catheterisation
        bowel_dys = predicted bowel dysfunction
        uri_hesitancy_weight = voting weight for uri_hesitancy
        uri_urgency_weight = voting weight for uri_urgency
        bladder_cath_weight = voting weight for bladder_cath
        bowel_dys_weight = voting weight for bowel_dys
    
    Outputs:
        Matrix of votes per each score
    """

    scores = get_scores_array()
    score_predictions = []
    for score in scores:
        predictions = []

        # check urinary hesitancy
        if uri_hesitancy == score[0]:
            predictions.append(uri_hesitancy_weight)
        else:
            predictions.append(0)
        
        # check urinary urgency
        if uri_urgency == score[1]:
            predictions.append(uri_urgency_weight)
        else:
            predictions.append(0)
        
        # check bladder catheter
        if bladder_cath == score[2]:
            predictions.append(bladder_cath_weight)
        else:
            predictions.append(0)

        # check bowel dysfunction
        if bowel_dys == score[3]:
            predictions.append(bowel_dys_weight)
        else:
            predictions.append(0)
        
        score_predictions.append(predictions)
    
    return score_predictions

def predict_urinary_hesitancy(note):
    """
    Obtain score for urinary hesitancy and retention
    - 0: none
    - 1: mild 
    - 2: moderate (urinary retention, frequent urinary tract infections)
    - 3: severe (catheterisation)
    - 4: loss of function (overflow incontinence)
    """

    p_urinary_hesitancy = re.compile(r"(?=)bladder|urinary|urine|urinary hesitancy|urinary retention|urine hesitancy|urine retention", re.IGNORECASE)
    p_mild = re.compile(r"(?=)mild|minimal|low|a bit|slight|not complet|issue", re.IGNORECASE)
    p_moderate = re.compile(r"(?=)moderate|medium|infection", re.IGNORECASE)
    p_severe = re.compile(r"(?=)severe|serious|extensive|significant", re.IGNORECASE)
    p_loss_of_function = re.compile(r"(?=)loss|no function|not functional", re.IGNORECASE)

    score = -1
    # if there is mention of urinary hesitancy / retention in note
    if len(re.findall(p_urinary_hesitancy, note)) > 0:
        # get urinary hesitancy level
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_urinary_hesitancy, sent)
            if mentions != None:
                # if mild 
                if len(re.findall(p_mild, sent)) > 0:
                    score = 1
                    break
                
                # if moderate 
                elif len(re.findall(p_moderate, sent)) > 0:
                    score = 2
                    break
                
                # if severe 
                elif len(re.findall(p_severe, sent)) > 0:
                    score = 3
                    break
                
                # if loss 
                elif len(re.findall(p_loss_of_function, sent)) > 0:
                    score = 4
                    break
                
                # if there was mention of urinary hesitancy but it wasn't picked up, guess none
                else:
                    score = 0
                    break

    return score

def predict_urinary_urgency(note):
    """
    Urinary urgency and incontinence
    - 0: none 
    - 1: mild 
    - 2: moderate (rare incontinence no more than 1 a week, wear pads)
    - 3: severe (frequent incontinence several times a week, urinal or pads )
    - 4: loss of function
    """
    p_urinary_urgency = re.compile(r"(?=)bladder|urine|urinary|urinary urgency|urinary incontinence|urine urgency|urine incontinence|incontinence", re.IGNORECASE)
    p_mild = re.compile(r"(?=)mild|minimal|low|slight|ongoing|complain|irrita|occassional|issue", re.IGNORECASE)
    p_moderate = re.compile(r"(?=)moderate|rare incontinence|pad|myrbetriq", re.IGNORECASE)
    p_severe = re.compile(r"(?=)severe|serious|extensive|significant|frequent|several|urinal|pads", re.IGNORECASE)
    p_loss_of_function = re.compile(r"(?=)loss|no function|not functional", re.IGNORECASE)

    score = -1
    # if there is mention of urinary urgency / incontinence in note
    if len(re.findall(p_urinary_urgency, note)) > 0:
        # get urinary hesitancy level
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_urinary_urgency, sent)
            if mentions != None:
                # if mild 
                if len(re.findall(p_mild, sent)) > 0:
                    score = 1
                    break
                
                # if moderate 
                elif len(re.findall(p_moderate, sent)) > 0:
                    score = 2
                    break
                
                # if severe 
                elif len(re.findall(p_severe, sent)) > 0:
                    score = 3
                    break
                
                # if loss 
                elif len(re.findall(p_loss_of_function, sent)) > 0:
                    score = 4
                    break
                
                # if there was mention of urinary hesitancy but it wasn't picked up, guess mild
                else:
                    score = 0
                    break
        
        # check incontinence
    return score

def predict_bladder_catheterisation(note):
    """
    Bladder catheterisation
    - 0: none
    - 1: intermittent self-catheterisation
    - 2: constant catheterisation
    """

    p_bladder_cath = re.compile(r"(?=)catheter|foley|iuc", re.IGNORECASE)
    p_intermittent = re.compile(r"(?=)intermittent|self", re.IGNORECASE)
    p_constant = re.compile(r"(?=)constant|always|indwelling|iuc", re.IGNORECASE)

    # default score
    score = -1
    # if there is mention of bladder catheterisation in note
    if len(re.findall(p_bladder_cath, note)) > 0:
        # get bladder catheterisation level
        sentences = sent_tokenize(note)
        for sent in sentences:
            mentions = re.search(p_bladder_cath, sent)
            if mentions != None:
                # if mild 
                if len(re.findall(p_intermittent, sent)) > 0:
                    score = 1
                    break
                
                # if moderate 
                elif len(re.findall(p_constant, sent)) > 0:
                    score = 2
                    break
                else:
                    score = 0
                    break
    return score

def predict_bowel_dysfunction(note):
    """
    Bowel dysfunction
    - 0: none
    - 1: mild: no incontinence, mild constipation
    - 2: moderate: must wear pads, alter lifestyle to be near lavatory
    - 3: severe: need enemata, manual measures to evacuate bowels
    - 4: loss of function
    """

    p_bowel_dys = re.compile(r"(?=)bowel|constipat|enema|stool|defecat", re.IGNORECASE)
    p_none = re.compile(r"(?=)no|none", re.IGNORECASE)
    p_mild = re.compile(r"(?=)no incontinence|mild|low|a bit|slight|increase", re.IGNORECASE)
    p_moderate = re.compile(r"(?=)moderate|pad", re.IGNORECASE)
    p_severe = re.compile(r"(?=)enema|severe|serious|evacuate", re.IGNORECASE)
    p_loss_of_function = re.compile(r"(?=)loss|no function|not functional", re.IGNORECASE)

    # default score
    score = -1 
    # if there is mention of bowel dysfunction in note
    if len(re.findall(p_bowel_dys, note)) > 0:
        # get bladder catheterisation level
        sentences = sent_tokenize(note)
        
        for sent in sentences:
            # TODO delete
            #sub_sentences = re.split(r"and|but|while\;|\,", sent) 
            #for sub_sent in sub_sentences:

            mentions = re.search(p_bowel_dys, sent)
            if mentions != None:

                # if mild 
                if len(re.findall(p_mild, sent)) > 0:
                    score = 1
                    break
                
                # if moderate 
                elif len(re.findall(p_moderate, sent)) > 0:
                    score = 2
                    break
                
                # if severe 
                elif len(re.findall(p_severe, sent)) > 0:
                    score = 3
                    break
                
                # if loss 
                elif len(re.findall(p_loss_of_function, sent)) > 0:
                    score = 4
                    break
                
                # else, guess normal
                else:
                    score = 0
    return score

@labeling_function()
def LF_bowel_bladder_original(df_row):
    """
    Calculate bowel bladder score based on Zhen's original rule-based code

    Inputs:
        df_row = 1 row of dataframe
    Output:
        bowel bladder score, -1 if unknown
    """

    note = df_row.text
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # Pattern
    p = re.compile(r"indwelling(?: Foley)? catheter|indwelling Foley", re.IGNORECASE)
    p2 = re.compile(r"Self-catheteriz|intermittent catheteriz|enema",re.IGNORECASE)
    
    # Prediction
    score = -1
    if edss_categorical == 0.0:
        score = 0
        return score
    
    elif len(re.findall(p, note)) > 0:
        score = 5
        return score
    
    elif len(re.findall(p2, note)) > 0:
        score = 3
        return score
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(r"( no real| not have).*(?:bladder control|bowel movement)", sent)) > 0:
            score = 4
            return score
    
    return score

@labeling_function()
def LF_bowel_bladder_average(df_row):
    """
    Predict bowel bladder score based on urinary hesitancy and retention, urinary urgency and incontinence, bladder catheterisation, bowel dysfunction

    Inputs:
        df_row = 1 row of dataframe
    Output:
        Gets max voted score that is closest to the average of max voted scores, -1 if unknown
    """

    note = df_row.text

    uri_hesitancy = predict_urinary_hesitancy(note)
    uri_urgency = predict_urinary_urgency(note)
    bladder_cath = predict_bladder_catheterisation(note)
    bowel_dys = predict_bowel_dysfunction(note)

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    score = -1

    if edss_categorical == 0:
        score = 0
        return 0

    # get matrix of possible scores given uri_hesitancy, uri_urgency, bladder_cath, bowel_dys
    scores = vote_scores(uri_hesitancy, uri_urgency, bladder_cath, bowel_dys, 1, 1, 1, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if uri_hesitancy, uri_urgency, bladder_cath, bowel_dys fits into at least 1
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

def get_bowel_bladder_lfs():

    return [LF_bowel_bladder_average, LF_bowel_bladder_original]

"""
RULES

Bowel and bladder functions depends on 

1. Urinary hesitancy and retention
- 0: none
- 1: mild 
- 2: moderate (urinary retention, frequent urinary tract infections)
- 3: severe (catheterisation)
- 4: loss of function (overflow incontinence)

2. Urinary urgency and incontinence
- 0: none 
- 1: mild 
- 2: moderate (rare incontinence no more than 1 a week, wear pads)
- 3: severe (frequent incontinence several times a week, urinal or pads )
- 4: loss of function

3. Bladder catheterisation
- 0: none
- 1: intermittent self-catheterisation
- 2: constant catheterisation

4. Bowel dysfunction
- 0: none
- 1: mild: no incontinence, mild constipation
- 2: moderate: must wear pads, alter lifestyle to be near lavatory
- 3: severe: need enemata, manual measures to evacuate bowels
- 4: loss of function

Self-catherize / intermittent catherization -> 3
Indwelling catheter / indwelling Foley catheter-> 4
5: complete loss of bowel_bladder function



SCORES
0: normal

1: 
- Urinary hesitancy and retention = 1
- Urinary urgency and incontinence = 1
- Bladder catheterisation = 0
- Bowel dysfunction = 0


2:
- Urinary hesitancy and retention = 2
- Urinary urgency and incontinence = 2
- Bladder catheterisation = 0
- Bowel dysfunction = 2

3: 
- Urinary hesitancy and retention = 4 (predict)
- Urinary urgency and incontinence = 3
- Bladder catheterisation = 1
- Bowel dysfunction = 3

4: 
- Urinary hesitancy and retention = 4 (predict)
- Urinary urgency and incontinence = 3 (carry)
- Bladder catheterisation = 2
- Bowel dysfunction = 3 (Carry)

5: 
- Urinary hesitancy and retention = 4 (predict)
- Urinary urgency and incontinence = 4
- Bladder catheterisation = 3
- Bowel dysfunction = 4 

Note: 6 is not used according to MS clinicians
Note: this subscore has lots of missingness

"""