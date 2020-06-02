import os
import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize



def predict(edss, note):
    

    # Patterns
    p_ambulation1 = re.compile(r"cane|pole", re.IGNORECASE)
    p_ambulation2 = re.compile(r"walker|rollator|two walking poles|two canes|two poles|2 walking poles|two trekking poles", re.IGNORECASE)


    # Set default Unknown
    score = -1
    
    # EDSS based general rule
    # EDSS 0 - Pyramidal 0
    if edss == 0.0:       
        score = 0
        return score
    
    if edss == 9.5:
        score = 15
        return score
    
    if edss == 9.0:
        score = 14
        return score

    if edss == 8.5:
        score = 13
        return score

    if edss == 8.0:
        score = 12
        return score

    if edss == 7.5:
        score = 11
        return score
    
    if edss == 7.0:
        score = 10
        return score

    if edss == 6.5:
        score = 8
        # Bilateral 9
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 9
        return score

    if edss == 6.0:
        score = 6
        # Bilateral 7
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 7
        return score
    
    if edss == 5.5:
        score = 4
        return score

    if edss == 5.0:
        score = 2
        # Bilateral 3
        if len(re.findall(p_ambulation2, note)) > 0:
            score = 3
        return score
    
    if edss == 4.5:
        score = 2
        return score
    
    if edss == 2.0:
        score = 1
        return score
    
    return score

"""
if __name__ == "__main__":
    df = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
    df = df.dropna()
    df = df.reset_index()

    predictions = []
    for i in range(df.shape[0]):
        predictions.append(score_ambulation_prediction(i, df))
    
    labels = np.array(df["score_ambulation_subscore"])
    labels[labels == -1] = 16
    
    # Classification Report
    print(classification_report(labels, np.array(predictions), digits = 4))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, np.array(predictions))
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    df_cm.columns = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","-1"]
    df_cm.rename(index = {0:"0",1:"1",2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10",11:"11",12:"12",13:"13",14:"14",15:"-1"}, inplace = True)
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
        elif predictions[i] != 16 and labels[i] != 16:
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