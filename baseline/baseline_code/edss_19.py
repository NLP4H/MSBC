import os
import pandas as pd
import re
from nltk.tokenize import word_tokenize, sent_tokenize
"""
# Sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# Plot
import seaborn
import matplotlib.pyplot as plt
"""


#target_names = ["0.0", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "5.5", "6.0", "6.5", "7.0", "7.5", "8.0", "8.5", "9.0", "Unknown"]

def predict(note):
    
    """
    Make predictions for overall EDSS score
    Input: one piece of note
    Output: EDSS class (e.g. 0 stands for 0.0, 18 stands for EDSS 9.5)
    
    """

    # Patterns for information extraction
    p = re.compile(r"edss", re.IGNORECASE)
    p_score = re.compile(r"\d\.\d")
    p_num = re.compile(r"zero|one|two|three|four|five|six|seven|eight|nine", re.IGNORECASE)
    num_dict = {
        "zero":0,
        "one":1,
        "two":2,
        "three":3,
        "four":4,
        "five":5,
        "six":6,
        "seven":7,
        "eight":8,
        "nine":9
    }
    score = -1
    sentences = sent_tokenize(note)
    for sent in sentences:
        # Find sentence with "EDSS"
        if len(re.findall(p, sent)) > 0:
            # Find score with format "x.x"
            if len(re.findall(p_score, sent)) > 0:
                score = float(re.findall(p_score, sent)[0])
                break
            # Find score with format "EDSS is x"
            elif len(re.findall(r"\s+(?:0|1|2|3|4|5|6|7|8|9)(?:\.|\,|\s+|\))", sent)) > 0:
                number = re.findall(r"\s+(?:0|1|2|3|4|5|6|7|8|9)(?:\.|\,|\s+|\))", sent)[0]
                score = float(re.sub(r"\s|\.|\,|\)", r"", number))
                break
            # Find score writtent in "zero/one ..."
            elif len(re.findall(p_num, sent)) > 0:
                score = float(num_dict[re.findall(p_num, sent)[0].lower()])
                break
    
    if score not in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]:
        score = -1
    
    
    label_dict = {0.0:0,
              1.0:1,
              1.5:2,
              2.0:3,
              2.5:4,
              3.0:5,
              3.5:6,
              4.0:7,
              4.5:8,
              5.0:9,
              5.5:10,
              6.0:11,
              6.5:12,
              7.0:13,
              7.5:14,
              8.0:15,
              8.5:16,
              9.0:17,
              9.5:18,
              -1:-1}
    
    return label_dict[score]


"""
if __name__ == "__main__":
    
    df_val = pd.read_csv("Z:/LKS-CHART/Projects/ms_clinic_project_new/ms_clinic_project/data/nlp_data/visit_level_data/valid_data.csv")
    df_val = df_val.dropna()
    df_val = df_val.reset_index()
    predictions = []
    for i in range(df_val.shape[0]):
        
        score = predict(df_val["text"][i])
        predictions.append(score)
    
    df_val["EDSS_predicted"] = predictions
    y_val = [label_dict[df_val["edss_19"][i]] for i in range(df_val.shape[0])]
    y_pred = [label_dict[df_val["EDSS_predicted"][i]] for i in range(df_val.shape[0])]
    print(classification_report(y_val, y_pred, target_names = target_names, digits=4))
    
    # Trouble shooting
    if y_val[i] == 0 and y_pred[i] != 0:
        print(i)
    

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    df_cm = pd.DataFrame(cm, range(cm.shape[0]), range(cm.shape[0]))
    df_cm.columns=target_names
    df_cm.rename(index={0:'0.0', 
                        1:'1.0',
                        2:'1.5',
                        3:'2.0',
                        4:'2.5', 
                        5:'3.0',
                        6:'3.5',
                        7:'4.0',
                        8:'4.5',
                        9:'5.0',
                        10:'5.5',
                        11:'6.0',
                        12:'6.5',
                        13:'7.0',
                        14:'7.5',
                        15:'8.0', 
                        16:'8.5', 
                        17:'9.0', 
                        18:'9.5', 
                        19:'Unknown'}, inplace=True)
    # Plot and save figure
    plt.figure(figsize = (10, 8))
    seaborn.set(font_scale = 1) # for label size
    ax = seaborn.heatmap(df_cm, annot = True, fmt = 'g', annot_kws = {"size": 13}) # font size
    ax.set(title = "Confusion Matrix for Multi-model Approach", xlabel = "Predicted EDSS", ylabel = "True EDSS")
    plt.xticks(rotation = 45)
    plt.yticks(rotation = 360)
    plt.savefig('confusion_matrix_keyword.png')
    plt.show()


    # Accuracy
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_val[i]:
            count += 1
    print("Accuracy: ", count / len(y_pred))

    # Converted Accuracy
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_val[i]:
            count += 1

        elif y_pred[i] != 19 and y_val[i] != 19 and abs(y_pred[i] - y_val[i]) <= 1:
            count += 1


    print("Converted Accuracy: ", count / len(y_pred))
"""


