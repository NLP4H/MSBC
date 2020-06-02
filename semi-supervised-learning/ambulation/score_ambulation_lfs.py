"""
Snorkel labelling functions for ambulation score
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
    Get matrix of possibles ranges of mobility_mode, edss_categorical, ambulation_distance for each ambulation score
    - Based on Neurostatus Definitions.pdf
    Output: matrix of possible ranges of mobility_mode, edss_categorical, ambulation_distance per score
    """

    scores= []
    score_0 = [[0], [0,9], [0, 1000]]
    score_1 = [[1], [3,9], [500, 100]]
    score_2 = [[1], [8,9], [300, 500]]
    score_3 = [[1], [9,9], [200, 300]]
    score_4 = [[1], [10,10], [100, 200]]
    score_5 = [[1], [11,11], [0, 100]]
    score_6 = [[2], [11,11], [50, 1000]]
    score_7 = [[3], [11,11], [120, 1000]]
    score_8 = [[2], [12,12], [0, 50]]
    score_9 = [[3], [12,12], [5, 120]]
    score_10 = [[4], [13,13], [0, 5]]
    score_11 = [[4], [14,14], [-10, -10]]   # -10 = no distance
    score_12 = [[4], [15, 18], [-10, -10]]  # -10 = no distance
    score_13 = [[4], [16, 16], [-10, -10]]
    score_14 = [[4], [17, 17], [-10, -10]]
    score_15 = [[4], [18, 18], [-10, -10]]
    scores = [score_0, score_1, score_2, score_3, score_4, score_5, score_6, score_7, score_8, score_9, score_10, score_11, score_12, score_13, score_14, score_15]
    
    return scores

def vote_scores(mobility_mode, edss_categorical, ambulation_distance, m_m_weight, e_weight, a_d_weight):
    """
    Given mobility, edss and distance prediction of a note --> vote on which scores the note could be
    Inputs:
        mobility_mode = predicted mobility mode
        edss_categorical = predicted edss_19 categorical 
        ambulation_distance = predicted ambulation distance
        m_m_weight = voting weight for mobility_mode
        e_weight = voting weight for edss_categorical
        ambulation_distance = voting weight for ambulation_distance
    
    Outputs:
        Matrix of votes per each score
    """

    scores = get_scores_array()
    score_predictions = []
    for score in scores:
        predictions = []

        # check mobility_mode
        if mobility_mode == score[0][0]:
            predictions.append(m_m_weight)
        else:
            predictions.append(0)
        
        # check edss 
        if edss_categorical >= score[1][0] and edss_categorical <= score[1][1]:
            predictions.append(e_weight)
        else:
            predictions.append(0)
        
        if ambulation_distance >= score[2][0] and ambulation_distance <= score[2][1]:
            predictions.append(a_d_weight)
        else:
            predictions.append(0)
        
        score_predictions.append(predictions)
    
    return score_predictions

def predict_ambulation_distance(note):
    """
    Given mention of ambulation, predict distance travelled
    Input: note in text format
    Output: ambulation distance, -1 if unknown
    """
    p_ambulation = re.compile(r"(?=)ambulat|walk|mobil|travel", re.IGNORECASE)
    p_int = re.compile(r"\d+")
    p_dec = re.compile(r"\d\.\d")
    distance = -1

    # get ambulation distance
    sentences = sent_tokenize(note)
    for sent in sentences:
        # Find sentence with ambulation related tokens
        ambulation_mentions = re.search(p_ambulation, sent)

        if ambulation_mentions != None:
            filtered_sentence = sent[ambulation_mentions.end():]

            # find value 
            if len(re.findall(p_int, filtered_sentence)) > 0:
                # get first number that is mentioned after the edss mention
                number = re.findall(p_int, filtered_sentence)[0]
                distance = float(re.sub(r"\s|\.|\=|\;|\-|\,|\)", r"", number))
                break

            # find score mentioned in decimal form
            elif len(re.findall(p_dec, filtered_sentence)) > 0:
                    # get first number that is mentioned after the edss mention
                distance = float(re.findall(p_dec, filtered_sentence)[0])
                break
            
            # non-sensical distance
            if distance < 0 or distance >= 1000:
                distance = -1
    
    return distance

def predict_mobility_mode(note):
    """
    Given mobility mode, predict mobility mode
    Input: note in text format
    Output: mobility mode, -1 if unknown
    
    mobility mode possibilities:
        -1 = unknown
        0 = unrestricted
        1 = p_no_assistance
        2 = unilateral assistance
        3 = bilateral assistance
        4 = restricted
    """

    p_unrestricted = re.compile(r"(?=)unrestrict", re.IGNORECASE)
    p_no_assistance = re.compile(r"(?=)without help|without assistance|without needing assistance|no help|no assistance|by himself|by herself|by themselves", re.IGNORECASE)
    p_assistance = re.compile(r"(?=)crutch|stick|brace|pole", re.IGNORECASE)
    p_restricted = re.compile(r"(?=)wheel|bed|constrain|transfer", re.IGNORECASE)

    mobility_mentions = [len(re.findall(p_unrestricted, note)), len(re.findall(p_no_assistance, note)), len(re.findall(p_assistance, note)), len(re.findall(p_restricted, note))]
    max_mobility_mention = max(mobility_mentions)
    # if there are no mobilty mentions
    if max_mobility_mention <= 0:
        return -1

    # assistance mention is highest / one of the highest = mobility mode is assistance
    elif max_mobility_mention == mobility_mentions[2]:
        assistance_mode = predict_assistance_mode(note)
        if assistance_mode == 1:
            return 2
        elif assistance_mode == 2:
            return 3
    
    # restricted mention is highest / one of the highest = mobility mode is restricted
    elif max_mobility_mention == mobility_mentions[3]:
        return 4

    # fully ambulatory mention is highest / one of the highest = mobility mode is fully ambulatory
    elif max_mobility_mention == mobility_mentions[1]:
        return 1
    
    # unrestricted mention is highest / one of the highest = mobility mode is unrestricted
    elif max_mobility_mention == mobility_mentions[0]:
        return 0

def predict_assistance_mode(note):
    """
    Given that predicted mobility mode is assistance, predict if it is unilateral or bilateral assistance
    Input: note in text format
    Output: assistance mode, -1 if unknown
    """

    p_assistance = re.compile(r"(?=)crutch|stick|brace|pole", re.IGNORECASE)
    p_bilateral_assistance = re.compile(r"(?=)2|two|double|bilateral", re.IGNORECASE)
    assistance_mode = 1

    # get assistant mode
    sentences = sent_tokenize(note)
    for sent in sentences:
        assistance_mentions = re.search(p_assistance, sent)

        # when assistance is mentioned: if they indicate bilateral assistance, update assistance_mode
        if assistance_mentions != None:
            if len(re.findall(p_bilateral_assistance, sent)) > 0:
                assistance_mode = 2

    return assistance_mode

# LABELLING FUNCTIONS

@labeling_function()
def LF_ambulation_average(df_row):
    """
    Predict ambulation score based mobility mode, edss and ambulation distance
    Gets max voted score that is closest to the average of max voted scores

    Inputs:
        df_row = 1 row of dataframe
    Outputs:
        ambulation score, -1 if unknown
    """
    note = df_row.text

     # get predicted mobility mode if available
        # eg. unrestricted, unilateral assistance, bilateral assistance, restricted
    mobility_mode = predict_mobility_mode(note)

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # get predicted ambulation distance if available
    ambulation_distance = predict_ambulation_distance(note)

    score = -1

    # get matrix of possible scores given mobilty mode, edss, ambulation distance
    scores = vote_scores(mobility_mode, edss_categorical, ambulation_distance, 1, 1, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mobility mode, edss or ambulation fits into at least 2 category
    if max_vote > 1:
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

@labeling_function()
def LF_ambulation_mobility_mode(df_row):
    """
    Predict ambulation score based mobility mode, edss and ambulation distannce
    Mobility mode is weighed 2x higher (ie. mobility mode considered the strongest predictor of ambulation)

    Inputs:
        df_row = 1 row of dataframe
    Output:
        Gets max voted score that is closest to the average of max voted scores, -1 if unknown
    """

    note = df_row.text

     # get predicted mobility mode if available
        # eg. unrestricted, unilateral assistance, bilateral assistance, restricted
    mobility_mode = predict_mobility_mode(note)

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # get predicted ambulation distance if available
    ambulation_distance = predict_ambulation_distance(note)

    score = -1

    # get matrix of possible scores given mobilty mode, edss, ambulation distance
        # mobility mode gets higher weight in voting
    scores = vote_scores(mobility_mode, edss_categorical, ambulation_distance, 1.5, 1, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mobility mode, edss or ambulation fits into at least 1 category
    if max_vote > 1:
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

@labeling_function()
def LF_ambulation_edss(df_row):
    """
    Predict ambulation score based mobility mode, edss and ambulation distannce
    edss is weighed 2x higher (ie. edss considered the strongest predictor of ambulation)

    Inputs:
        df_row = 1 row of dataframe
    Output:
        Gets max voted score that is closest to the average of max voted scores, -1 if unknown
    """
    note = df_row.text

     # get predicted mobility mode if available
        # eg. unrestricted, unilateral assistance, bilateral assistance, restricted
    mobility_mode = predict_mobility_mode(note)

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # get predicted ambulation distance if available
    ambulation_distance = predict_ambulation_distance(note)

    score = -1

    # get matrix of possible scores given mobilty mode, edss, ambulation distance
        # edss gets higher weight in voting
    scores = vote_scores(mobility_mode, edss_categorical, ambulation_distance, 1, 1.5, 1)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mobility mode, edss or ambulation fits into at least 1 category
    if max_vote > 1:
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

@labeling_function()
def LF_ambulation_distance(df_row):
    """
    Predict ambulation score based mobility mode, edss and ambulation distannce
    Ambulation distance is weighed 2x higher (ie. ambulation distance considered the strongest predictor of ambulation)

    Inputs:
        df_row = 1 row of dataframe
    Output:
        Gets max voted score that is closest to the average of max voted scores, -1 if unknown
    """
    note = df_row.text
     # get predicted mobility mode if available
        # eg. unrestricted, unilateral assistance, bilateral assistance, restricted
    mobility_mode = predict_mobility_mode(note)

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    # get predicted ambulation distance if available
    ambulation_distance = predict_ambulation_distance(note)

    score = -1

    # get matrix of possible scores given mobilty mode, edss, ambulation distance
        # ambulation distance  gets higher weight in voting
    scores = vote_scores(mobility_mode, edss_categorical, ambulation_distance, 1, 1, 1.5)
    scores = np.asarray(scores)

    # get max prediction
    vote_per_score = np.sum(scores, axis = 1)
    max_vote = np.max(vote_per_score)

    # if mobility mode, edss or ambulation fits into at least 1 category
    if max_vote > 1:
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

@labeling_function()
def LF_ambulation_original(df_row):
    """
    Calculate ambulation based on Zhen's original rule-based code

    Inputs:
        df_row = 1 row of dataframe
    Output:
        ambulation score, -1 if unknown
    """
    note = df_row.text
    
    score = -1  # default to -1 if not found
    
    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1
    
    # Patterns
    p_ambulation1 = re.compile(r"cane|pole", re.IGNORECASE)
    p_ambulation2 = re.compile(r"walker|rollator|two walking poles|two canes|two poles|2 walking poles|two trekking poles", re.IGNORECASE)


    # Set default Unknown
    score = -1
    
    # EDSS based general rule
    # EDSS 0 - Pyramidal 0

    # edss = 0 --> edss_categorical = 0
    if edss_categorical == 0:       
        score = 0
        return score
    
    # TODO replace this
    # edss = 9.5 --> edss_categorical = 18
    if edss_categorical == 18:
        score = 15
        return score

    # edss = 9 --> edss_categorical = 17
    if edss_categorical == 17:
        score = 14
        return score

    # edss = 8.5 --> edss_categorical = 16
    elif edss_categorical == 16:
        score = 13
        return score

    # edss = 8.0 --> edss_categorical = 15
    if edss_categorical == 15:
        score = 12
        return score

    # edss = 7.5 --> edss_categorical = 14
    if edss_categorical == 14:
        score = 11
        return score
    
    # edss = 7.0 --> edss_categorical = 13
    if edss_categorical == 13:
        score = 10
        return score

    # edss = 6.5 --> edss_categorical = 12
    if edss_categorical == 12:
        score = 8
        # Bilateral 9
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 9
        return score

    # edss = 6.0 --> edss_categorical = 11
    if edss_categorical == 11:
        score = 6
        # Bilateral 7
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 7
        return score
    
    # edss = 5.5 --> edss_categorical = 10
    if edss_categorical == 10:
        score = 4
        return score

    # edss = 5.0 --> edss_categorical = 9
    if edss_categorical == 9:
        score = 2
        # Bilateral 3
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 3
        return score
    
    # edss = 4.5 --> edss_categorical = 8
    if edss_categorical == 8:
        score = 2
        return score
    
    # edss = 2.0 --> edss_categorical = 3
    if edss_categorical == 3:
        score = 1
        return score
    
    return score

def get_ambulation_lfs():
    # Uncomment when you want to test just original rules
    # return [LF_ambulation_original]

    # Uncomment to test just new LFs
    # return [LF_ambulation_average, LF_ambulation_mobility_mode, LF_ambulation_edss, LF_ambulation_distance]

    return [LF_ambulation_average, LF_ambulation_mobility_mode, LF_ambulation_edss, LF_ambulation_distance, LF_ambulation_original]


# RULES
# ambulation depends on edss, mobility modes, and ambulation distance

# edss score 
    # 0 - 18

# mobility mode:
    # -1 = unknown
    # 0 = unrestricted
    # 1 = p_no_assistance
    # 2 = unilateral assistance
    # 3 = bilateral assistance
    # 4 = restricted

# score = 0 
    # mobility = 0 
    # edss = 0 - 5  --> edss_categorical 0-9
    # distance = 0-1000

# score = 1
    # mobility = 1
    # edss = 2 - 5 --> edss_categorical 3-9
    # distance = 500-100

# score = 2
    # mobility = 1
    # edss = 4.5 - 5 --> edss_categorical 8-9
    # distance = 300-500

# score = 3
    # mobility = 1
    # edss = 5 --> edss_categorical 9
    # distance = 200 - 300

# score = 4
    # mobility = 1
    # edss = 5.5 --> edss_categorical 10
    # distance = 100 - 200

# score = 5
    # mobility = 1
    # edss = 6 --> edss_categorical 11
    # distance = 0 - 100

# score = 6
    # mobility = 2
    # edss = 6 --> edss_categorical 11
    # distance = 50 - 1000

# score = 7 
    # mobility = 3
    # edss = 6 --> edss_categorical 11
    # distance = 120 - 1000

# score = 8
    # mobility = 2
    # edss = 6.5 --> edss_categorical 12
    # distance = 0 - 50

# score = 9
    # mobility = 3
    # edss = 6.5 --> edss_categorical 12
    # distance = 5 - 120

# score = 10
    # mobility = 4
    # edss = 7 --> edss_categorical 13
    # distance = 0-5

# score 11
    # mobility = 4
    # edss = 7.5 --> edss_categorical 14
    # distance = -1 

# score = 12
    # mobility = 4
    # edss = 8 --> edss_categorical 15
    # distance = -1 