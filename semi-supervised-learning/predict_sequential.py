import pandas as pd
import numpy as np
import re
import os
import sys
import pdb

def prediction(labels):
    """
    if rule found predict what the rule found otherwise predict from model output
    """
    y_pred = []
    for i in labels:
        rb = i[0]
        cnn = i[1]
        if (rb==-1):
            y_pred.append(cnn)
        else:
            y_pred.append(rb)
    
    return np.asarray(y_pred)

# save predictions to csv using standard rule base approach from baseline
def predict_sequential(label_type, df_result_path, prediction_col_names, model_type):   
    """
    input: df path, prediction col names, and model_type
    output: saved .csv
    saves .csv in format: patient_id|date|split|text|edss_19|predicted_edss_19_<model_type>
    """
    # pdb.set_trace()
    print("reading in df")
    # drop nans in text and labels (i.e. -1)
    df = pd.read_csv(df_result_path)
    empty_text = df[df.text.isna()]
    df = df.dropna(subset = ['text'])

    df_unlabeled = df[df.split=='unlabeled']
    df = df[df.split!='unlabeled']

    df[label_type] = df[label_type].fillna(-1)
    empty_labels = df[df[label_type]==-1]
    df = df[df[label_type] != -1]

    # format [[label_1,label_2,]]
    labels = df[prediction_col_names].to_numpy()

    labels_pred = df_unlabeled[prediction_col_names].to_numpy()


    # fill prediction for empty text entries with -1
    empty_text["predict_" + label_type + model_type] = -1
    empty_labels["predict_" + label_type + model_type] = -1

    #predict and save file
    print("predicting")
    print("model_type: " + model_type)
    predict = prediction(labels)
    df["predict_" + label_type + model_type] = predict
    predict_unlabeled = prediction(labels_pred)
    df_unlabeled["predict_" + label_type + model_type] = predict_unlabeled


    # keep only columns we are concerned with
    print("keeping only columns we are concerned with")
    df = df[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    df_unlabeled = df_unlabeled[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    empty_text = empty_text[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    empty_labels = empty_labels[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    #concatenate all dataframes
    print("concat")
    df_result = pd.concat([df, df_unlabeled, empty_text, empty_labels], ignore_index=True)

    # save final dataframe result
    print("saving")
    save_dir = "master_path/results/task1/"
    output_name = "predict_" + label_type + model_type +".csv"
    os.chdir(save_dir)
    df_result.to_csv(output_name)
    print("shape of final dataframe")
    print(df_result.shape)
    print("done :)")

    return

# #uncomment below to run 
# result_path = "master_path/results/task1/edss_19_results.csv"
# label_type = ""
# prediction_col = ['predict_' + label_type + 'rb_baseline','predict_' + label_type + 'word2vec_cnn'] #predicted_edss_19_allen_cnn, predict_edss_19_word2vec_cnn
# model_t = "rb_word2vec_cnn" #rb_word2vec_cnn, rb_allen_cnn
# predict_sequential(result_path,prediction_col,model_t)   