import pandas as pd
import numpy as np
import os
from os import path
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pdb
 
#report on F1 scores for different prediction types in results.csv
def report_acc(prediction_col_name):
    """
    inputs:
        - prediction column name
    outputs: 
        - sklearn F1 scores 
    """
    # read dataframe
    results_csv_path = "master_path/results/task1/edss_19_results.csv"
    print("reading results")
    df = pd.read_csv(results_csv_path)
    # drop nans in text and labels (i.e. -1)
    df = df.dropna(subset = ['text'])
    df = df[df.edss_19 != -1]

    # report train accuracy
    df_train = df[df.split=='train']
    y_true = df_train['edss_19']
    y_pred = df_train[prediction_col_name]

    # print(classification_report(y_true,y_pred))

    acc = accuracy_score(y_true,y_pred)
    f1_macro = f1_score(y_true,y_pred, average = 'macro')
    f1_micro = f1_score(y_true,y_pred, average = 'micro')
    f1_weighted = f1_score(y_true,y_pred, average = 'weighted')

    print("training set results")
    # print("training accuracy: %.2f" %acc)
    print("f1 macro: %.2f"  %f1_macro)
    print("f1 micro: %.2f"  %f1_micro)
    print("f1 weighted: %.2f" %f1_weighted)

    # report val accuracy
    df_val = df[df.split=='val']
    y_true = df_val['edss_19']
    y_pred = df_val[prediction_col_name]

    # print(classification_report(y_true,y_pred))

    acc = accuracy_score(y_true,y_pred)
    f1_macro = f1_score(y_true,y_pred, average = 'macro')
    f1_micro = f1_score(y_true,y_pred, average = 'micro')
    f1_weighted = f1_score(y_true,y_pred, average = 'weighted')

    print("validation set results")
    # print("val accuracy: %.2f" %acc)
    print("f1 macro: %.2f"  %f1_macro)
    print("f1 micro: %.2f"  %f1_micro)
    print("f1 weighted: %.2f" %f1_weighted)

    # report test accuracy
    df_test = df[df.split=='test']
    y_true = df_test['edss_19']
    y_pred = df_test[prediction_col_name]

    # print(classification_report(y_true,y_pred))
    
    acc = accuracy_score(y_true,y_pred)
    f1_macro = f1_score(y_true,y_pred, average = 'macro')
    f1_micro = f1_score(y_true,y_pred, average = 'micro')
    f1_weighted = f1_score(y_true,y_pred, average = 'weighted')
    
    print("test set results")
    # print("testing accuracy: %.2f" %acc)
    print("f1 macro: %.2f"  %f1_macro)
    print("f1 micro: %.2f"  %f1_micro)
    print("f1 weighted: %.2f" %f1_weighted)

    return

# F1 micro, F1 macro, accuracy

##### MODEL PREDICTIONS MADE THUS FAR ####
# predict_edss_19_word2vec_cnn
# predicted_edss_19_allen_cnn
# predict_edss_19_log_reg_baseline
# predict_edss_19_linear_svc
# predict_edss_19_svc_rbf
# predict_edss_19_lda
# predict_edss_19_rb_baseline
# predict_edss_19_rb_allen_cnn
# predict_edss_19_rb_word2vec_cnn
# predict_edss_19_snorkel_rb_allen_cnn_lm
# predict_edss_19_snorkel_rb_word2vec_cnn_lm
# predict_edss_19_snorkel_rb_tfidf_word2vec_cnn_lm
# predict_edss_19_snorkel_rb_tfidf_allen_cnn_lm
###########################################
report_acc("predict_edss_19_snorkel_rb_tfidf_allen_cnn_lm")
