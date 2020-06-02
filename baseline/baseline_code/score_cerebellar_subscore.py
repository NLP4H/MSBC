import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize



def check_heel_to_shin(note):
    print("Heel-to-shin")

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
                print("severe")
                break
            # mild/minimal 2
            elif len(re.findall(p_mild, sent)) > 0:
                score = 2
                print("mild")
                break             
            # reasonable 1
            elif len(re.findall(r"reasonabl", sent)) > 0:
                score = 1
                print("reasonable")
                break
            elif len(re.findall(r"assess", sent)) > 0:
                score = 2
                print("cannot be assessed")
                break
            elif len(re.findall(r"moderate|Moderate|impair|Impair|difficult", sent)) > 0:
                score = 3
                print("moderate")
                break
    print(score)
    return score

def check_ataxia(note):
    print("ataxia/dysmetria")
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
            print("severe")
            break
        if len(re.findall(p_mild, sent)) > 0:
            print("mild")
            score = 2
            break
        if len(re.findall(p2, sent)) > 0:
            score = 2
            break
        


    print(score)
    return score

def check_tremor(note):
    print("tremor")
    score = -1
    p = re.compile(r"intention tremor|postural tremor", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0:
            if len(re.findall(p, sent)) > 0:
                score = 1
                print("1")
                break
    print(score)
    return score

def check_tandem_gait(note):
    print("tandem gait")
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
                print("mild")
                break
            elif len(re.findall(r"not able to perform|unable to perform|unable to do|cannot perform|could not perform|could not be performed", sent)) > 0:
                score = 3
                print("can't perform")
                break
            elif len(re.findall(p_neg, sent)) > 0:
                score = 0
                print("negation 0")
                break
            elif len(re.findall(r"impair|difficult", sent)) > 0:
                score = 2
                print("moderate/impaired")
                break
    print(score)
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
    
    
    print("General Rule: ")
    score = -1
    p = re.compile(r"(?:neurological|neurologic)? exam", re.IGNORECASE)
    sentences = sent_tokenize(note)
    for sent in sentences:
        if len(re.findall(p, sent)) > 0 and len(re.findall(r"normal", sent)) > 0:
            score = 0
            break
    print(score)
    return score
        
def predict(edss, note):
    
    # EDSS = 0
    if edss == 0.0:       
        score = 0
        return score
    
    # EDSS != 0
    else:
        selected_note = select_neuro_exam(note)
        score = max(general_rule(selected_note), check_ataxia(selected_note), check_tremor(selected_note), check_heel_to_shin(selected_note), check_tandem_gait(selected_note))
        if score == 4 and edss < 4.0:
            score = 3
        if score == 4 and edss <= 3.0:
            score = 2        
        return score


           

"""
if __name__ == "__main__":
    df = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
    df = df.dropna()
    df = df.reset_index()

    predictions = []
    labels = np.array(df["score_cerebellar_subscore"])
    labels[labels == -1] = 6
    for i in range(df.shape[0]):    
        predictions.append(score_cerebellar_prediction(i, df))



    labels = np.array(df["score_cerebellar_subscore"])
    labels[labels == -1] = 6
    predictions = np.array(predictions)
    predictions[predictions == -1] = 6
    # Classification Report
    print(classification_report(labels, predictions, digits = 4))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    df_cm.columns = ["0","1","2","3","4","-1"]
    df_cm.rename(index = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"-1"}, inplace = True)
    plt.figure(figsize = (10, 8))
    sn.set(font_scale = 1) # for label size
    sn.heatmap(df_cm, annot = True, fmt = 'g', annot_kws = {"size": 15}) # font size
    plt.show()

    # Converted Accuracy
    wrong_predictions = []
    count = 0
    
    for i in range(len(labels)):

        if predictions[i] == labels[i]:
            count += 1
        elif predictions[i] != -1 and labels[i] != -1:
            if abs(labels[i] - predictions[i]) <= 1:
                count += 1
        else:
            wrong_predictions.append(i)

    converted_acc = count / len(labels)
    print("Converted Accuracy: ", converted_acc)

    # Accuracy
    count = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            count += 1
    print("Accuracy: ", count / len(labels))

"""

