import pandas as pd
import numpy as np
import re
import os
import sys

from edss_19_lfs import get_snorkel_compatible_lfs
sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning/')
import snorkel_functions
from snorkel_functions import get_label_predictions
import load_data
from load_data import convert_edss_19_to_categorical

def prediction(df):
    """
    sequentially look through rule base and add label found
    """

    # EDSS_19 Labelling
    lfs = get_snorkel_compatible_lfs()
    label_type = "edss_19"

    labels = get_label_predictions(df,lfs,"")
    y_pred = []
    for i in labels:
        rb = i[0:3]
        if np.all(rb==-1):
            y_pred.append(-1)
        else:
            #int
            if(rb[0]!=-1):
                y_pred.append(rb[0])
            #dec
            elif(rb[1]!=-1):
                y_pred.append(rb[1])
            #word
            elif(rb[2]!=-1):
                y_pred.append(rb[2])
            #roman
            else:
                y_pred.append(rb[3])
    
    return np.asarray(y_pred)

# save predictions to csv using standard rule base approach from baseline
def predict_rb(df_train_path, df_val_path, df_test_path, df_unlabeled_path, model_type):   
    """
    input: df path for train,test,val,unlabeled, model_type (the name)
    output: saved .csv
    saves .csv in format: patient_id|date|split|text|edss_19|predicted_edss_19_<model_type>
    """

    print("reading in df")
    #read in dataframes 
    df_train = pd.read_csv(df_train_path)
    df_val = pd.read_csv(df_val_path)
    df_test = pd.read_csv(df_test_path)
    df_unlabeled= pd.read_csv(df_unlabeled_path)


    # create split column and denote if dataframe is train, test, val, or unlabeled
    print("split column added")
    df_train['split'] = "train"
    df_val['split'] = "val"
    df_test['split'] = "test"
    df_unlabeled['split'] = "unlabeled"

    # drop na's in text to predict edss_label. observations with empty text will be predicted with label -1 (not always the case)
    df_train[['edss_19']] = df_train.edss_19.fillna(-1)
    empty_text_train = df_train.loc[np.where(df_train.text.isna())]
    df_train = df_train.dropna(subset = ['text'])

    df_val[['edss_19']] = df_val.edss_19.fillna(-1)
    empty_text_val = df_val.loc[np.where(df_val.text.isna())]
    df_val = df_val.dropna(subset = ['text'])
    
    df_test[['edss_19']] = df_test.edss_19.fillna(-1)
    empty_text_test = df_test.loc[np.where(df_test.text.isna())]
    df_test = df_test.dropna(subset = ['text'])

    df_unlabeled[['edss_19']] = df_unlabeled.edss_19.fillna(-1)
    empty_text_unlabeled = df_test.loc[np.where(df_test.text.isna())]
    df_unlabeled = df_unlabeled.dropna(subset = ['text'])
    
    # fill prediction for empty text entries with -1
    empty_text_train['predict_edss_19_' + model_type] = -1
    empty_text_val['predict_edss_19_'  + model_type] = -1
    empty_text_test['predict_edss_19_' + model_type] = -1
    empty_text_unlabeled['predict_edss_19_' + model_type] = -1

    # convert edss_19 to categorical from 0 - 18
    print("converting edss to categorical")
    df_train['edss_19'] = convert_edss_19_to_categorical(df_train)
    df_val['edss_19'] = convert_edss_19_to_categorical(df_val)
    df_test['edss_19'] = convert_edss_19_to_categorical(df_test)
    df_unlabeled['edss_19'] = convert_edss_19_to_categorical(df_unlabeled)

    # convert score 19 to -1
    print("modyfying edss 19 to -1")
    df_train['edss_19'] = df_train['edss_19'].replace(19, -1)
    df_val['edss_19'] = df_val['edss_19'].replace(19, -1)
    df_test['edss_19'] = df_test['edss_19'].replace(19, -1)
    df_unlabeled['edss_19'] = df_unlabeled['edss_19'].replace(19, -1)
    


    #predict and save file
    print("predicting train")
    print("model_type: "+ model_type)
    predict_train = prediction(df_train)
    df_train['predict_edss_19_'+model_type] = predict_train
    print("predicting val")
    predict_val = prediction(df_val)
    df_val['predict_edss_19_'+model_type] = predict_val
    print("predicting test")
    predict_test = prediction(df_test)
    df_test['predict_edss_19_'+model_type] = predict_test
    print("predicting unlabeled")
    predict_unlabeled = prediction(df_unlabeled)
    df_unlabeled['predict_edss_19_'+model_type] = predict_unlabeled

    # keep only columns we are concerned with
    print("keeping only columns we are concerned with")
    df_train = df_train[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    df_val = df_val[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    df_test = df_test[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    df_unlabeled = df_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    
    empty_text_train = empty_text_train[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_text_val = empty_text_val[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_text_test = empty_text_test[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_text_unlabeled = empty_text_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]

    #concatenate all dataframes
    print("concat")
    df = pd.concat([df_train, df_val, df_test, df_unlabeled, empty_text_train, empty_text_val, empty_text_test, empty_text_unlabeled], ignore_index=True)

    # save final dataframe result
    print("saving")
    save_dir = "master_path/results/task1/"
    output_name = "predict_edss_19_"+model_type+".csv"
    os.chdir(save_dir)
    df.to_csv(output_name)
    print("shape of final dataframe")
    print(df.shape)
    print("done :)")
    
    return
# Uncomment below to run
# train_path = "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv"
# val_path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
# test_path = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
# unlabeled_path = "master_path/data/neurology_notes/processed_data/Final Splits/unlabeled_data.csv"
# model_t = "rb_baseline" 
# predict_rb(train_path,val_path,test_path,unlabeled_path,model_t)    
    