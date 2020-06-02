import pandas as pd
import numpy as np
import re
import os
import sys

# import labelling functions
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/edss_19/')
from edss_19_lfs import get_snorkel_compatible_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/ambulation/')
from score_ambulation_lfs import get_ambulation_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/bowel_bladder/')
from score_bowel_bladder_lfs import get_bowel_bladder_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/brain_stem/')
from score_brain_stem_lfs import get_brain_stem_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/cerebellar/')
from score_cerebellar_lfs import get_cerebellar_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/pyramidal/')
from score_pyramidal_lfs import get_pyramidal_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/sensory/')
from score_sensory_lfs import get_sensory_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/visual/')
from score_visual_lfs import get_visual_lfs
sys.path.append('master_path//ML4H_MSProject/semi-supervised-learning/mental/')
from score_mental_lfs import get_mental_lfs

import snorkel_functions
from snorkel_functions import get_label_predictions
import load_data
from load_data import convert_edss_19_to_categorical

def prediction(df, label_type):
    """
    sequentially look through rule base and add label found
    """
    # Labelling
    if label_type == "edss_19":
        lfs = get_snorkel_compatible_lfs()
    
    elif label_type == "score_ambulation_subscore":
        lfs = get_ambulation_lfs()
    
    elif label_type == "score_bowel_bladder_subscore":
        lfs = get_bowel_bladder_lfs()

    elif label_type == "score_brain_stem_subscore":
        lfs = get_brain_stem_lfs()

    elif label_type == "score_cerebellar_subscore":
        lfs = get_cerebellar_lfs()

    elif label_type == "score_mental_subscore":
        lfs = get_mental_lfs()

    elif label_type == "score_pyramidal_subscore":
        lfs = get_pyramidal_lfs()

    elif label_type == "score_sensory_subscore":
        lfs = get_sensory_lfs()

    elif label_type == "score_visual_subscore":
        lfs = get_visual_lfs()

    else:
        print(label_type + " not found") 
       
    # Save rules
    labels = get_label_predictions(df,lfs,"")
    save_dir = "master_path//repo/ML4H_MSProject/data/baseline_data/"
    os.chdir(save_dir)
    np.savez(label_type + "labels" + ".npz", labels)

    # load rules
    # save_dir = "master_path//repo/ML4H_MSProject/data/baseline_data/"
    # os.chdir(save_dir)
    # npzfile = np.load(save_dir + label_type + "labels" + ".npz", labels)
    # labels = npzfile['arr_0'] 
    
    y_pred = []
    for i in labels:
        if np.all(i==-1):
            y_pred.append(-1)
        else:
            for j in i:
                if j!=-1:
                    y_pred.append(j)
                    break
    return np.asarray(y_pred)

# save predictions to csv using standard rule base approach from baseline
def predict_rb(label_type, df_train_path, df_val_path, df_test_path, df_unlabeled_path, model_type):   
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

    # convert edss_19 to categorical from 0 - 18
    print("converting edss to categorical")
    df_train['edss_19'] = df_train.edss_19.fillna(-1)
    df_train['edss_19'] = convert_edss_19_to_categorical(df_train)
    df_val['edss_19'] = df_val.edss_19.fillna(-1)
    df_val['edss_19'] = convert_edss_19_to_categorical(df_val)
    df_test['edss_19'] = df_test.edss_19.fillna(-1)
    df_test['edss_19'] = convert_edss_19_to_categorical(df_test)
    df_unlabeled['edss_19'] = df_unlabeled.edss_19.fillna(-1)
    df_unlabeled['edss_19'] = convert_edss_19_to_categorical(df_unlabeled)

    # drop na's in text to predict edss_label. observations with empty text will be predicted with label -1 (not always the case)
    df_train[[label_type]] = df_train[label_type].fillna(-1)
    empty_text_train = df_train.loc[np.where(df_train.text.isna())]
    df_train = df_train.dropna(subset = ['text'])

    df_val[[label_type]] = df_val[label_type].fillna(-1)
    empty_text_val = df_val.loc[np.where(df_val.text.isna())]
    df_val = df_val.dropna(subset = ['text'])
    
    df_test[[label_type]] = df_test[label_type].fillna(-1)
    empty_text_test = df_test.loc[np.where(df_test.text.isna())]
    df_test = df_test.dropna(subset = ['text'])

    df_unlabeled[[label_type]] = df_unlabeled[label_type].fillna(-1)
    empty_text_unlabeled = df_test.loc[np.where(df_test.text.isna())]
    df_unlabeled = df_unlabeled.dropna(subset = ['text'])
    
    # fill prediction for empty text entries with -1
    empty_text_train["predict_" + label_type + model_type] = -1
    empty_text_val["predict_" + label_type + model_type] = -1
    empty_text_test["predict_" + label_type + model_type] = -1
    empty_text_unlabeled["predict_" + label_type + model_type] = -1

    #predict and save file
    print("predicting train")
    print("model_type: "+ model_type)
    predict_train = prediction(df_train, label_type)
    df_train["predict_" + label_type + model_type] = predict_train
    print("predicting val")
    predict_val = prediction(df_val, label_type)
    df_val["predict_" + label_type + model_type] = predict_val
    print("predicting test")
    predict_test = prediction(df_test, label_type)
    df_test["predict_" + label_type + model_type] = predict_test
    print("predicting unlabeled")
    predict_unlabeled = prediction(df_unlabeled, label_type)
    df_unlabeled["predict_" + label_type + model_type] = predict_unlabeled

    # keep only columns we are concerned with
    print("keeping only columns we are concerned with")
    df_train = df_train[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    df_val = df_val[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    df_test = df_test[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    df_unlabeled = df_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    
    empty_text_train = empty_text_train[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    empty_text_val = empty_text_val[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    empty_text_test = empty_text_test[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]
    empty_text_unlabeled = empty_text_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', label_type, "predict_" + label_type + model_type]]

    #concatenate all dataframes
    print("concat")
    df = pd.concat([df_train, df_val, df_test, df_unlabeled, empty_text_train, empty_text_val, empty_text_test, empty_text_unlabeled], ignore_index=True)

    # save final dataframe result
    print("saving")
    save_dir = "master_path/results/task1/"
    output_name = "predict_" + label_type + model_type+".csv"
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
# label_t = "score_ambulation_subscore"
# model_t = "_rb_baseline" 
# predict_rb(label_t, train_path,val_path,test_path,unlabeled_path,model_t)    
    