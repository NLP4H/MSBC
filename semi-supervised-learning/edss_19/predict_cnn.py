# predict functions (workaround for snorkel and torch incompatibility issues)
import pandas as pd
import numpy as np
import os
from os import path
import sys

# will need to use GPU if training model file - uncomment line below
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import re
import tensorflow as tf
from keras.models import load_model
import pickle
import cnn_word2vec_helperfunctions
from cnn_word2vec_helperfunctions import notes_to_sequences
sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning/')
from load_data import convert_edss_19_to_categorical
import pdb


###########
## PREDICT FUNCTIONS
###########
   
# prediction for cnn model -> conda enviornemnt issues with tensorflow-gpu and snorkel version 0.93
def main(argv): 
    """
    Predict edss_19 using a cnn model. CNN model uses trained Word2Vec embeddings. 
    Input: 
        df_path
    Output: EDSS class (0-18) for all observations
    """
    path = str(argv[1])
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    # Fill NA values in labels to -1 and drop them
    df = df.fillna(-1)
    df = df[df.edss_19 != -1]
    text_df = df['text'].to_frame()

    # paths for saved model and tokenizer
    model_path = "master_path//repo/ML4H_MSProject/data/baseline_models/"
    tokenizer_path = "master_path//repo/ML4H_MSProject/data/baseline_data/"    
    
    # load cnn model
    model = load_model(model_path +"edss_19.h5")

    #predict
    # model predicts from text_to_sequence input
    # thus, remove punctuation and make sure lower case. Tokenize and remove stop words. pad sequence to target input
    notes= []
    for i,row in text_df.iterrows():
        note = cnn_word2vec_helperfunctions.notes_to_sequences(row['text'], tokenizer_path)
        notes.append(note[0])        
    notes=np.asarray(notes)

    score = model.predict(notes)
    y_pred_class = score.argmax(axis = -1)
    #replace 19 with -1 for snorkel
    y_pred_class = np.where(y_pred_class==19,-1, y_pred_class)

    for i in y_pred_class:
        sys.stdout.write(str(i)+' ')

main(sys.argv)

# this is called for returning numpy array taking a df (not a path)
def prediction(df):
    """
    returns np.array based on cnn prediction for text
    """
    # paths for saved model and tokenizer
    model_path = "master_path//repo/ML4H_MSProject/data/baseline_models/"
    tokenizer_path = "master_path//repo/ML4H_MSProject/data/baseline_data/"    
    
    # load cnn model
    model = load_model(model_path +"edss_19.h5")

    #predict
    # model predicts from text_to_sequence input
    # thus, remove punctuation and make sure lower case. Tokenize and remove stop words. pad sequence to target input
    notes= []
    for i,row in df.iterrows():
        note = cnn_word2vec_helperfunctions.notes_to_sequences(row['text'], tokenizer_path)
        notes.append(note[0])
    notes=np.asarray(notes)

    #predict
    score = model.predict(notes)
    y_pred_class = score.argmax(axis = -1)
    #replace 19 with -1 for snorkel
    y_pred_class = np.where(y_pred_class==19,-1, y_pred_class)

    return y_pred_class

# save cnn prediction result on all current notes to dataframe 
def predict_cnn(df_train_path, df_val_path, df_test_path, df_unlabeled_path):
    """
    input: df path for train,test,val,unlabeled
    output: saved .csv
    saves .csv in format: patient_id|date|split|text|edss_19|predicted_edss_19_word2vec_cnn
    """
    # pdb.set_trace()
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
    empty_labels_train = df_train[df_train.edss_19==-1]
    df_train = df_train[df_train.edss_19 != -1]

    df_val[['edss_19']] = df_val.edss_19.fillna(-1)
    empty_text_val = df_val.loc[np.where(df_val.text.isna())]
    df_val = df_val.dropna(subset = ['text'])
    empty_labels_val = df_val[df_val.edss_19==-1]
    df_val = df_val[df_val.edss_19 != -1]
    
    df_test[['edss_19']] = df_test.edss_19.fillna(-1)
    empty_text_test = df_test.loc[np.where(df_test.text.isna())]
    df_test = df_test.dropna(subset = ['text'])
    empty_labels_test = df_test[df_test.edss_19==-1]
    df_test = df_test[df_test.edss_19 != -1]

    df_unlabeled[['edss_19']] = df_unlabeled.edss_19.fillna(-1)
    empty_text_unlabeled = df_test.loc[np.where(df_test.text.isna())]
    df_unlabeled = df_unlabeled.dropna(subset = ['text'])
    
    # fill prediction for empty text entries with -1
    empty_text_train['predict_edss_19_word2vec_cnn'] = -1
    empty_text_val['predict_edss_19_word2vec_cnn'] = -1
    empty_text_test['predict_edss_19_word2vec_cnn'] = -1
    empty_text_unlabeled['predict_edss_19_word2vec_cnn'] = -1
    empty_labels_train['predict_edss_19_word2vec_cnn'] = -1
    empty_labels_val['predict_edss_19_word2vec_cnn'] = -1
    empty_labels_test['predict_edss_19_word2vec_cnn'] = -1

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
    predict_train = prediction(df_train)
    df_train['predict_edss_19_word2vec_cnn'] = predict_train
    print("predicting val")
    predict_val = prediction(df_val)
    df_val['predict_edss_19_word2vec_cnn'] = predict_val
    print("predicting test")
    predict_test = prediction(df_test)
    df_test['predict_edss_19_word2vec_cnn'] = predict_test
    print("predicting unlabeled")
    predict_unlabeled = prediction(df_unlabeled)
    df_unlabeled['predict_edss_19_word2vec_cnn'] = predict_unlabeled

    # keep only columns we are concerned with
    print("keeping only columns we are concerned with")
    df_train = df_train[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    df_val = df_val[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    df_test = df_test[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    df_unlabeled = df_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    
    empty_text_train = empty_text_train[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_text_val = empty_text_val[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_text_test = empty_text_test[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_text_unlabeled = empty_text_unlabeled[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_labels_train = empty_labels_train[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_labels_val = empty_labels_val[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]
    empty_labels_test = empty_labels_test[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_word2vec_cnn']]

    #concatenate all dataframes
    print("concat")
    df = pd.concat([df_train, df_val, df_test, df_unlabeled, empty_text_train, empty_text_val, empty_text_test, empty_text_unlabeled, empty_labels_train, empty_labels_val, empty_labels_test], ignore_index=True)

    # save final dataframe result
    print("saving")
    save_dir = "master_path/results/task1/"
    output_name = "predict_edss_19_word2vec_cnn.csv"
    os.chdir(save_dir)
    df.to_csv(output_name)
    print("shape of final dataframe")
    print(df.shape)
    print("done :)")
    return


# uncomment below to run - dont forget to comment main
# train_path = "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv"
# val_path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
# test_path = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
# unlabeled_path = "master_path/data/neurology_notes/processed_data/Final Splits/unlabeled_data.csv"

# predict_cnn(train_path,val_path,test_path,unlabeled_path)