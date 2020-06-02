import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize




def predict(edss, note):
    

    p = re.compile(r"subjective cognitive complaints", re.IGNORECASE)
    p2 = re.compile(r"Montreal Cognitive Assessment|MoCA", re.IGNORECASE)
    p3 = re.compile(r"(?:mild|Mild) cognitive challenge", re.IGNORECASE)
    p4 = re.compile(r"(?:mild|Mild) fatigue", re.IGNORECASE)
    p_neg = re.compile(r"No | no | deni|not have|not had", re.IGNORECASE)

    # Unknown by default
    score = -1
    
    if edss == 0:
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
        if edss == 0.0:
            score = 1
        elif edss > 3.0:
            score = 3
        else:
            score = 2

        
    if len(re.findall(p, note)) > 0:
        score = 1
        if edss == 2.0:
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


"""

if __name__ == "__main__":


    df = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
    df = df.dropna()
    df = df.reset_index()


    #labels = [convert_score(df["score_visual_subscore"][i]) for i  in range(df.shape[0])]
    labels = np.array(df["score_mental_subscore"])
    #labels[labels == -1] = 6
    predictions = []
    for i in range(df.shape[0]): 

        #print(i) 
        #print(df["score_mental_subscore"][i], score_mental_prediction(i,df))
        predictions.append(score_mental_prediction(i, df))



    labels = np.array(labels)
    labels[labels == -1] = 6
    predictions = np.array(predictions)
    predictions[predictions == -1] = 6
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