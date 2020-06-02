import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize


    
def predict(edss, note):
    
    
    # Pattern
    p = re.compile(r"indwelling(?: Foley)? catheter|indwelling Foley", re.IGNORECASE)
    p2 = re.compile(r"Self-catheteriz|intermittent catheteriz|enema",re.IGNORECASE)
    
    # Prediction
    score = -1
    if edss == 0.0:
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


"""
if __name__ == "__main__":

    df = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
    df = df.dropna()
    df = df.reset_index()

    predictions = []
    labels = np.array(df["score_bowel_bladder_subscore"])
    labels[labels == -1] = 7
    for i in range(df.shape[0]):    
        predictions.append(score_bowel_bladder_prediction(i, df))
        
        
        # Look at patient's previous subscores if Unknown
        if predictions[i] == 7:
            patient_id = df["patient_id"][i]
            visit_date = df["visit_date"][i]
            df_selected = df[df["patient_id"] == df["patient_id"][i]]
            possible_scores = df_selected[df_selected["visit_date"] < visit_date]["score_bowel_bladder_subscore"]
            if len(possible_scores) != 0:
                predictions[i] = max(possible_scores)

                
    predictions = np.array(predictions)
    predictions[predictions == -1] = 7

    print(classification_report(labels, predictions, digits=4))
    cm = confusion_matrix(labels, predictions)
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    df_cm.columns = ["0","1","2","3","4","5","-1"]
    df_cm.rename(index = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"-1"}, inplace = True)
    plt.figure(figsize = (10, 8))
    sn.set(font_scale = 1) # for label size
    sn.heatmap(df_cm, annot = True, fmt = 'g', annot_kws = {"size": 15}) # font size
    plt.show()

    # accuracy
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count += 1
    print("Accuracy: ", count / len(predictions))

    # Converted accuracy
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            count += 1
        
        elif predictions[i] != 7 and labels[i] != 7 and abs(predictions[i] - labels[i]) <= 1:
            count += 1
    print("Converted Accuracy: ", count / len(predictions))

"""