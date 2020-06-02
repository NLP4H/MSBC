# predict functions (workaround for snorkel and torch incompatibility issues)
import pandas as pd
import numpy as np
import os
from os import path
import sys

# will need to use GPU if training model file - uncomment line below
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score

import re
import pickle

import pdb

sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning')
import load_data
from load_data import convert_edss_19_to_categorical
import tfidf_helperfunctions
from tfidf_helperfunctions import tokenize_neurology

###########
## PREDICT FUNCTIONS
###########
   
# prediction for tfidf model -> conda enviornemnt issues with snorkel version 0.93
def main(argv):
    """
    Predict edss_19 using a tfidf model.  
    Input: 
        df_path, model_type
    Output: EDSS class (0-18) for all observations
    """
    #input load
    path = str(argv[1])
    model_type = str(argv[2])
    # path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
    # model_type="linear_svc"
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    # Fill NA values in labels to -1
    df = df.fillna(-1)
    df = df[df.edss_19 != -1]
    text_df = df['text'].to_frame()

    #train load for vectorizer
    #TODO: save vectorizer in set up and load here
    label_type = "edss_19"

    #get notes into list
    texts = list(text_df['text'])
    #make type numpy
    texts = np.array(texts)
    
    # paths for saved model and tokenizer
    model_path = "master_path//repo/ML4H_MSProject/data/baseline_models/"   
    
    # load tfidfvectorizer and transform
    os.chdir(model_path)
    model_name = "tf.pkl"
    with open(model_name, 'rb') as file:
        tf = pickle.load(file)
    # tfidf transform
    X = tf.transform(texts)


    # load model
    os.chdir(model_path)
    if model_type == "log_reg_baseline":
        with open("log_reg_baseline.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "lda":
        with open("lda.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "svc_rbf":
        with open("svc_rbf.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "linear_svc":
        with open("linear_svc.pkl",'rb') as file:
            model = pickle.load(file)

    else:
        print("not yet implemented")

    #predict
    # model predicts from tfidf input X
    if model_type == "lda":
        score = model.predict(X.todense())
    else:
        score = model.predict(X)
    # y_pred_class = score.argmax(axis = -1)
    #replace 19 with -1 for snorkel
    y_pred_class = np.where(score==19,-1, score)

    for i in y_pred_class:
        sys.stdout.write(str(i)+' ')

main(sys.argv)


# this is called for returning numpy array taking a df (not a path)
def prediction( df, model_type):
    """
    returns np.array based on tfidf prediction for df containing 'text'
    """
    #train load for vectorizer
    label_type = "edss_19"

    #get notes into list
    texts = list(df['text'])
    #make type numpy
    texts = np.array(texts)
    
    # paths for saved model and tokenizer
    model_path = "master_path//repo/ML4H_MSProject/data/baseline_models/"   
    
    # load tfidfvectorizer and transform
    os.chdir(model_path)
    model_name = "tf.pkl"
    with open(model_name, 'rb') as file:
        tf = pickle.load(file)
    # tfidf transform
    X = tf.transform(texts)


    # load model
    os.chdir(model_path)
    if model_type == "log_reg_baseline":
        with open("log_reg_baseline.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "lda":
        with open("lda.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "svc_rbf":
        with open("svc_rbf.pkl", 'rb') as file:
            model = pickle.load(file)

    elif model_type == "linear_svc":
        with open("linear_svc.pkl",'rb') as file:
            model = pickle.load(file)

    else:
        print("not yet implemented")

    #predict
    # model predicts from tfidf input X
    if model_type == "lda":
        score = model.predict(X.todense())
    else:
        score = model.predict(X)
    # y_pred_class = score.argmax(axis = -1)
    #replace 19 with -1 for snorkel
    y_pred_class = np.where(score==19,-1, score)

    return y_pred_class

# save tfidf prediction result on all current notes to dataframe 
def predict_tfidf(df_train_path, df_val_path, df_test_path, df_unlabeled_path, model_type):
    """
    input: df path for train,test,val,unlabeled, model type
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
    empty_text_train['predict_edss_19_' + model_type] = -1
    empty_text_val['predict_edss_19_'  + model_type] = -1
    empty_text_test['predict_edss_19_' + model_type] = -1
    empty_text_unlabeled['predict_edss_19_' + model_type] = -1
    empty_labels_train['predict_edss_19_'+model_type] = -1
    empty_labels_val['predict_edss_19_'+model_type] = -1
    empty_labels_test['predict_edss_19_'+model_type] = -1

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
    predict_train = prediction(df_train, model_type)
    df_train['predict_edss_19_'+model_type] = predict_train
    print("predicting val")
    predict_val = prediction(df_val, model_type)
    df_val['predict_edss_19_'+model_type] = predict_val
    print("predicting test")
    predict_test = prediction(df_test, model_type)
    df_test['predict_edss_19_'+model_type] = predict_test
    print("predicting unlabeled")
    predict_unlabeled = prediction(df_unlabeled, model_type)
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
    empty_labels_train = empty_labels_train[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_labels_val = empty_labels_val[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_labels_test = empty_labels_test[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]

    #concatenate all dataframes
    print("concat")
    df = pd.concat([df_train, df_val, df_test, df_unlabeled, empty_text_train, empty_text_val, empty_text_test, empty_text_unlabeled, empty_labels_train, empty_labels_val, empty_labels_test], ignore_index=True)

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


# uncomment below to run predict_tfidf - dont forget to comment main
# train_path = "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv"
# val_path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
# test_path = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
# unlabeled_path = "master_path/data/neurology_notes/processed_data/Final Splits/unlabeled_data.csv"
# model_t = "lda" #"log_reg_baseline", "svc_rbf", "lda", "linear_svc"
# predict_tfidf(train_path,val_path,test_path,unlabeled_path,model_t)