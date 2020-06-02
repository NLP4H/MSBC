import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize




def convert_score(raw_score):
    """
    Convert raw visual score to converted score
    raw_score = 1, converted_score = 1
    raw_score = 2, converted_score = 2
    raw_score = 3, converted_score = 2
    raw_score = 4, converted_score = 3
    raw_score = 5, converted_score = 3
    raw_score = 6, converted_score = 4
    raw_score = -1, converted_score = -1
    
    Input: raw visual score
    Return: converted visual score

    """
    convert_score = -1
    if raw_score in [0,1,2]:
        converted_score = raw_score
    elif raw_score in [3,4]:
        converted_score = raw_score - 1
    elif raw_score in [5,6]:
        converted_score = raw_score - 2
    # Unknown
    elif raw_score == -1:
        converted_score = -1
    return converted_score

def check_visual_acuity(note):

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
                    print("Visual Acuity: ", score)
                    return score

        print("worst:", worst_eye)
        print("best:", best_eye)
        # 20/20 normal
        if worst_eye == 20:
            score = 0
            print("Visual Acuity: ", score)
            return score
        

        # 1: 20/20 – 20/30 
        elif worst_eye > 20 and worst_eye <= 30:
            score = 1
            print("Visual Acuity: ", score)
            return score
        # 2: 20/30 – 20/60  
        elif worst_eye > 30 and worst_eye <= 60:
            score = 2
            print("Visual Acuity: ", score)
            return score
        # 3: 20/60 – 20/100 
        elif worst_eye > 60 and worst_eye <= 100:
            score = 3
            print("Visual Acuity: ", score)
            return score
        # 4: 20/100 – 20/200 
        elif (worst_eye > 100 and worst_eye <= 200) or \
            (worst_eye != best_eye and worst_eye > 60 and worst_eye <= 100 and best_eye > 60 and best_eye <= 100):
            score = 4
            print("Visual Acuity: ", score)
            return score
        # 5: > 200
        elif (worst_eye > 200) or \
            (worst_eye != best_eye and worst_eye > 100 and worst_eye <= 200 and best_eye > 60 and best_eye <= 200):
            score = 5
            print("Visual Acuity: ", score)
            return score
            
        # 6: worst eye > 200, best eye >= 60
        elif (worst_eye > 200):
            score = 6
            print("Visual Acuity: ", score)
            return score
        
        
    # "Visual acuity" is detected
    elif len(re.findall(p2, note)) > 0:
        sentences = sent_tokenize(note)
        for sent in sentences:
            if len(re.findall(p2, sent)) > 0 and len(re.findall(r"normal|Normal", sent)) > 0:
                score = 0
                print("Visual Acuity: ", score)
                return score
    
    print("Visual Acuity: ", score)
    return score

def check_pallor(note):
    
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
    
    print("Pallor:", score)
    return score

def check_scotoma(note):
    
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
            print(sent)

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
    
    print("Scotoma: ", score)
    return score

def check_visual_fields(note):
    
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

    print("Visual Fields: ", score)
    return score

def general_rule(note):

    """
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
            print("No visual symptons")
            score = 0
            break
        if len(re.findall(r"neurological exam", sent)) > 0 and len(re.findall(r"normal", sent)) > 0:
            print("Neurological exam normal")
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
            print("Blind/Finger counting")
            score = 6
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"foot", sent)) > 0:
            print("Finger counting 1 ft")
            score = 5
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"2 feet|two feet|3 feet|three feet", sent)) > 0:
            print("Finger counting 2/3 ft")
            score = 4
            break
        elif len(re.findall(r"finger counting", sent)) > 0 and len(re.findall(r"light perception", sent)) > 0:
            print("Finger counting & light perception")
            score = 6
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s4", sent)) > 0 and len(re.findall(r"vision alone", sent)) > 0:
            print("EDSS 4 related to vision")
            score = 6
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s3", sent)) > 0 and len(re.findall(r"vision|visual sign", sent)) > 0:
            print("EDSS 3 related to vision")
            score = 4
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s2", sent)) > 0 and len(re.findall(r"vision|visual sign", sent)) > 0:
            score = 2
            break
        elif len(re.findall(r"EDSS", sent)) > 0 and len(re.findall(r"\s4", sent)) > 0 and len(re.findall(r"loss of vision", sent)) > 0:
            print("EDSS 4 related to vision")
            score = 4
            break
        phrases = sent.split(",")
        for phrase in phrases:
            if len(re.findall(r"vision|visual", phrase)) > 0 and len(re.findall(r"significant", phrase)) > 0 and len(re.findall(r"impair", phrase)) > 0:
                if len(re.findall(r"improv", note)) > 0:
                    break

                score = 6
                break
    print("General Rule: ", score)
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

def predict(edss, note):
    
    # Unknown by default
    score = -1
    selected_note = select_neuro_exam(note)

    # EDSS = 0 all scores 0
    if edss == 0:
        score = 0

    else:
        score = max(general_rule(selected_note), check_visual_acuity(selected_note), check_pallor(selected_note), check_scotoma(selected_note), check_visual_fields(selected_note))
        # Convert score
        # score = convert_score(score)

    return score


"""
df = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
df = df.dropna()
df = df.reset_index()

predictions = []
#labels = [convert_score(df["score_visual_subscore"][i]) for i  in range(df.shape[0])]
labels = np.array(df["score_visual_subscore"])
#labels[labels == -1] = 6


# Trouble Shooting

for i in range(df.shape[0]): 

    print(i) 
    predictions.append(score_visual_prediction(i, df))
    
    

# Deal with Unknowns
# Method: Check other possible scores for the same patient
final_predictions = []
for i in range(df.shape[0]):
    
    if predictions[i] == -1:
        possible_scores = list(df[df["patient_id"] == df["patient_id"][i]]["score_visual_subscore"])
        if list(set(possible_scores)) == [-1]:
            final_predictions.append(-1)
        else:
            final_predictions.append(max(set(possible_scores), key = possible_scores.count))
    else:
        final_predictions.append(predictions[i])



# Evaluation
labels[labels == -1] = 7
final_predictions = np.array(final_predictions)
final_predictions[final_predictions == -1] = 7
# Classification Report
print(classification_report(labels, final_predictions, digits = 4))

"""