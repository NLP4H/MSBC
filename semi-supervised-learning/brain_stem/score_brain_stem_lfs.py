"""
Snorkel labelling functions for brain stem score
Note: This brain stem subscore is not based on Neurostatus defns but 
based on heureustic information provided by Zhen based on advice given from MS clinicians
"""

import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
import snorkel
from snorkel.labeling import labeling_function

def predict_eom(note):
    
    score = -1
    # Patterns
    p = re.compile(r"extraocular movement(s)?", re.IGNORECASE)

    p_neg1 = re.compile(r"extraocular movement(s)?(\s+)?(of the eye |and saccades )?(were |are |essentially |\, |seem )?(full|normal|intact|complete)|(intact |full |normal |unremarkable )(including )?(visual fields( and |\, | as well as |, and ))?extraocular movements",re.IGNORECASE)
    p_neg2 = re.compile(r"smooth pursuit|no extraocular movement (restriction|abnormalities)", re.IGNORECASE)
    
    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:

            # Saccadic pursuit
            if len(re.findall(r"(saccadic|choppy) pursuit", sent))  > 0 and \
                len(re.findall(r"no (saccadic|choppy) pursuit", sent)) == 0:
                score = 1
                break

            # Normal - 0
            if len(re.findall(p_neg1, sent)) > 0 or \
                (len(re.findall(p_neg2, sent)) > 0 and len(re.findall(r"broken", sent)) == 0):
               score = 0
               break
                
            if len(re.findall(r"significant lateral deviation", sent)) > 0:
                score = 3
                break
            
        if len(re.findall(r"psychotic intrusions", sent)) or \
        len(re.findall(r"cannot abduct (her|his) (left|right) eye|abduction deficit|(a|A)bduction of the (right|left) eye ", sent)) > 0 or \
        len(re.findall(r"(left |right )nasolabial fold flattening", sent)) > 0:
            score = 2
            break

    # # print("Eye movements: ", score)
    return score

def predict_INO(note):

    # Unknown by default
    score = -1
    # Patterns
    p = re.compile(r"INO|internuclear ophthalmoplegia")
    p_neg = re.compile(r"((no|without) (evidence of |frank |appreciable )|no |nor |without (any )?|did not (show|demonstrate|seen|have) |not able to detect any )((nystagmus|RAPD)( or (evidence of )?(an )?|\, )|definite )?(an )?(INO|(internuclear )?ophthalmoplegia)",re.IGNORECASE)
    p1 = re.compile(r"(subtle )?((b|B)ilateral |(r|R)ight |(l|L)eft )INO|((s|S)ubtle )(bilateral |right |left )?INO")
    
    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:

            # Normal - 0
            if len(re.findall(p_neg, sent)) > 0:
               score = 0
               break
            
            if len(re.findall(p1, sent)) > 0:
                score = 1
                break
                
            
            
    # # print("INO: ", score)
    return score

def predict_nystagmus(note):
    
    score = -1
    p = re.compile(r"nystagmus", re.IGNORECASE)
    p_neg = re.compile(r"(no |nor )nystagmus", re.IGNORECASE)
    p1 = re.compile(r"broken smooth pursuit|horizontal nystagmus bilaterally|has( mild)? nystagmus( on ends)?|(end\-gaze|coarse|rotatory|show(s)?) nystagmus|gaze(\-|\s)(evoked|invoked) (horizontal )?nystagmus", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                
        if len(re.findall(p1, sent)) > 0:
            if len(re.findall(r"(Mild|mild) nystagmus", sent)) > 0:
                score = 1
                break
            score = 2
            break
    # print("Nystagmus:", score)
    return score

def predict_palate(note):
    score = -1
    p = re.compile(r"palate", re.IGNORECASE)
    p_neg = re.compile(r"Symmetrical palate raise|palate (rises|elevates) symmetrically", re.IGNORECASE)
    
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
    # print("Palate:", score)
    return score

def predict_tongue(note):
    
    score = -1
    p = re.compile(r"tongue", re.IGNORECASE)
    p_neg = re.compile(r"no tongue deviation|Tongue (was )?midline", re.IGNORECASE)
    p1 = re.compile(r"(minor|mild) clumsiness of tongue movement", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            
            # Normal
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            
            # Abnormal (mild)
            if len(re.findall(p1, sent)) > 0:
                score = 2
                break
    # print("Tongue: ", score)
    return score

def predict_pupillary_response(note):
    
    # Unknown by default
    score = -1
    
    # Patterns
    p = re.compile(r"pupil|(APD|(relative )?afferent (pupillary )?defect)", re.IGNORECASE)
    p_neg_rapd = re.compile(r"not (have|has|demonstrate|appreciate) (an )?(INO or |disc pallor or )?(RAPD|(relative )?afferent (pupillary )?defect)|(did not detect |No |no |without |not |(No|no) evidence of )(clear |note(d)? |INO or |disc pallor or |nystagmus or )?(an |the )?(RAPD|(relative )?afferent (pupillary )?defect)")
    p_neg2 = re.compile(r"(pupillary |pupil(s)? )(reaction(s)? |restriction )?(is |was |were |are )?(not )?(reactive|normal|equal|repeated|re\-examined)|(reactive |normal )(pupillary reflexes|pupils)", re.IGNORECASE)
    p2 = re.compile(r"((s|S)he|(h|H)e has |a clear) RAPD|(left(\-sided)? |right(\-sided)? |(detect|has|with) (an |a subtle )?|mild )((R)?APD|(relative )?afferent (pupillary )?defect)")
    
    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:

            # Normal - 0
            if len(re.findall(p_neg_rapd, sent)) > 0 or \
                len(re.findall(p_neg2, sent)) > 0:
               score = 0
            
            # Abnormal - 1
            if len(re.findall(p2, sent)) > 0:
                score = 1
                break
    # print("Pupillary Reaction: ", score)
    return score
# TODO: For future use
def predict_trigeminal_damage(note):
    
    score = -1
    # print("Trigeminal damage: ", score)
    return score

def predict_ptosis(note):
    # Unknown by default
    score = -1
    
    # Patterns
    p_neg = re.compile(r"(No|minimal)( left|right|left\-sided|right\-sided)? ptosis", re.IGNORECASE)
    p1 = re.compile(r"(mild |chronic )(right(\-sided)? |left(\-sided)? |bilateral |non\-fatigable )?(eyelid |upper lid )?ptosis", re.IGNORECASE)
    p2 = re.compile(r"(?:(right|left|bilateral|non\-fatigable)(\-sided)?)?( eyelid| upper lid)? ptosis", re.IGNORECASE)
    
    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(r"ptosis|Ptosis", sent)) > 0:

            # No facial weakness
            if len(re.findall(p_neg, sent)) > 0:
                score = 0
                break
            
            # Mild
            elif len(re.findall(p1, sent)) > 0:
                score = 1
                break
           
            # Moderate
            elif len(re.findall(p2, sent)) > 0 and \
                len(re.findall(r"mild|Mild|minimal|Minimal", sent)) == 0:
                score = 1
                break
            
    # print("Ptosis: ", score)
    return score

def predict_facial_weakness(note):
    
    # Unknown by default
    score = -1
    
    # Patterns
    p = re.compile(r"facial|sensation|palsy", re.IGNORECASE)
    p_neg1 = re.compile(r"Facial\s+((sensation(s)?|motor|strength|musculature|movement(s)?)\s+)(?:and (sensory( and examination(s)?)?|sensation|strength|motor|power)\s+)?(?:is |was |are |were )?(?:normal|full|intact|equal|symmetric)", re.IGNORECASE)
    p_neg2 = re.compile(r"(?:denies |no |not (?:had|has) |no evidence of )(?:obvious |definite |notable )?facial (?:weakness|droop|numbness|pain|asymmetry)", re.IGNORECASE)
    p_neg3 = re.compile(r"(?:normal|intact)\s+facial\s+(?:sensation|symmetry|motor|strength|musculature|movement(s)?)", re.IGNORECASE)
    
    p2 = re.compile(r"(?:has|left|right|bilateral|motor neuron)\s+facial\s+(numbness|pain|weakness|palsy)", re.IGNORECASE)
    p3 = re.compile(r"(diminished |decreased )(?:left |right )?(?:facial )?(light touch and temperature )?sensation|Facial sensation.*(minimal |mildly |diminished )((impairment of|impaired) light touch|temperature)", re.IGNORECASE)
    p4 = re.compile(r"(has|had) a (left |right )sixth nerve palsy|shows a partial left sixth nerve|mild (left |right )6th nerve palsy")
    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:

            # No facial weakness
            if len(re.findall(p_neg1, sent)) > 0 or \
                len(re.findall(p_neg2, sent)) > 0 or \
                    len(re.findall(p_neg3, sent)) > 0:
                score = 0
            
            # Abnormalities
            if len(re.findall(p2, sent)) > 0:
                score = 1
                break
            
        # Diminished facial sensation
        if (len(re.findall(p3, sent)) > 0 and len(re.findall(r"(p|P)inprick|dermatome|no |thumb|arm|leg|finger|foot|toes|hand|(lower|upper) (limbs|extremities)|hemibody", sent)) == 0) or \
        len(re.findall(p4, sent)) > 0 or \
        (len(re.findall(r"sixth nerve", sent)) > 0 and len(re.findall(r"blurring", sent)) > 0) or \
        (len(re.findall(r"lip", sent)) > 0 and len(re.findall(r"tingling", sent)) > 0) or \
        (len(re.findall(r"(f|F)acial sensation", sent)) > 0 and len(re.findall(r"diminished", sent)) > 0):
            # print("detected")
            score = 2
            break


    # print("Facial weakness: ", score)
    return score

def predict_hearing_loss(note):
    
    # Unknown by default
    score = -1
    # Patterns
    p_neg = re.compile(r"hearing\s+(?:was|is)?(\s+)?(?:grossly )?(?:fine|normal|intact)", re.IGNORECASE)

    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        # Hearing is intact
        if len(re.findall(p_neg, sent)) > 0:
            score = 0
            break
    
    # print("Hearing:", score)
    return score

def predict_dysarthria(note):
    
    # Unknown by default
    score = -1
    # Patterns
    p = re.compile(r"dysarthri", re.IGNORECASE)
    p_neg = re.compile(r"(no (evidence of |clear )?|not convinced of )(?:definite )?dysarthria|denies dysarthria|no history of (?:dysphagia or )?dysarthria", re.IGNORECASE)
    p_minimal = re.compile(r"(?:slight|minimal|mild scanning ),?(?:\s+)?(?:if any)?,?(?:\s+)?(?:spastic )?dysarthria", re.IGNORECASE)
    p_mild = re.compile(r"(?:mild(ly)?),?(?:\s+)?(?:if any)?,?(?:\s+)?(?:spastic )?(dysarthria|dysarthric)", re.IGNORECASE)
    p_severe = re.compile(r"significant dysarthria", re.IGNORECASE)
    p_moderate = re.compile(r"ha(s|d) a spastic dysarthria|(?:clear|has|evidence of) dysarthria|moderate dysarthria|dysarthria was noted", re.IGNORECASE)

    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p_neg, sent)) > 0:
            score = 0
            break
        elif len(re.findall(p_neg, sent)) == 0 and len(re.findall(p, sent)) > 0:
            # Minimal/Signs only
            if len(re.findall(p_minimal, sent)) > 0:
                score = 1
                break
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                score = 2
                break
            # Moderate
            if len(re.findall(p_moderate, sent)) > 0:
                score = 3
                break
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 4
                break
    
    # print("Dysarthria:", score)
    return score

def predict_dysphagia(note):
    
    score = -1
    # Patterns
    p = re.compile(r"dysphagia", re.IGNORECASE)
    p_neg = re.compile(r"no (?:definite )?dysphagia|denies dysphagia|no history of (?:dysarthria or )?dysphagia", re.IGNORECASE)
    p_minimal = re.compile(r"(?:slight|minimal|occasional)(?:\s+)dysphagia", re.IGNORECASE)
    p_mild = re.compile(r"(?:mild)(?:\s+)dysphagia", re.IGNORECASE)
    p_severe = re.compile(r"significant dysphagia", re.IGNORECASE)
    p_moderate = re.compile(r"(?:clear|has|evidence of)? dysphagia|moderate dysphagia|dysphagia was noted", re.IGNORECASE)

    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p_neg, sent)) > 0:
            score = 0
            break
        elif len(re.findall(p_neg, sent)) == 0 and len(re.findall(p, sent)) > 0:
            # Minimal/Signs only
            if len(re.findall(p_minimal, sent)) > 0:
                score = 1
                break
            # Mild
            if len(re.findall(p_mild, sent)) > 0:
                score = 2
                break
            # Moderate
            if len(re.findall(p_moderate, sent)) > 0:
                score = 3
                break
            # Severe
            if len(re.findall(p_severe, sent)) > 0:
                score = 4
                break
    
    # print("Dysphagia:", score)
    return score

def general_rules(note):
    
    # Unknown by default
    score = -1
    # Pattern
    p1 = re.compile(r"cranial nerves", re.IGNORECASE)
    p2 = re.compile(r"lower|apart from|other|remainder|remaining|although|rest|but|with the exception of|except for", re.IGNORECASE)
    p3 = re.compile(r"III|IV| V |VI|VII|VIII|IX| X |XI")
    p_neg = re.compile(r"normal|intact|unremarkable", re.IGNORECASE)

    # Predictions
    sentences = sent_tokenize(note)
    for sent in sentences:

        # Normal cranial nerves
        if len(re.findall(p1, sent)) > 0 and \
            len(re.findall(p_neg, sent)) > 0 and \
                len(re.findall(p2, sent)) == 0 and \
                    len(re.findall(p3, sent)) == 0:
                    score = 0
                    break

    # print("General Rule: ", score)
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
def LF_brain_stem_original(df_row):
    
    note = df_row.text

    if "edss_19" in np.asarray(df_row.index):
        edss_categorical = df_row.edss_19
    else:
        edss_categorical = -1

    score = -1
    selected_note = select_neuro_exam(note)

    if edss_categorical == 0:
        score = 0
    
    elif general_rules(selected_note) == 0:
        score = 0

    else:
        score = max(
            predict_eom(selected_note),
            predict_INO(selected_note), 
            predict_nystagmus(selected_note),
            predict_palate(selected_note),
            predict_tongue(selected_note),
            predict_pupillary_response(selected_note),
            predict_trigeminal_damage(selected_note),
            predict_ptosis(selected_note),
            predict_facial_weakness(selected_note),
            predict_hearing_loss(selected_note),
            predict_dysarthria(selected_note),
            predict_dysphagia(selected_note))
    return score

def get_brain_stem_lfs():
    return [LF_brain_stem_original]