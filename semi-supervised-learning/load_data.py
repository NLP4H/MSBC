# Load train, valid and raw data for labelling
# TODO: fillna? 
import pandas as pd
import numpy as np
import os
# TODO: more conversion functions for subscores
def convert_edss_19_to_categorical(labels_df):
    '''
    Convert edss_19 score from dataset into 19 categorical variables
    Inputs: df: dataframe of train, test or raw data
    '''
    edss_original = labels_df['edss_19'].values
    # Labels converted to int categories
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
    
    edss_categorical = [label_dict[edss] for edss in edss_original]

    return edss_categorical

def get_raw_df():
    '''
    Get raw / unlabelled neurology notes (preprocessed by de-id script)
    Outputs: 
        text_df: dataframe of notes
    '''
    path = "master_path/data/neurology_notes/processed_data/neurology_notes_processed.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    text_df = df['text'].to_frame()

    return text_df, path

# TODO: implement get_train_df for other label_type (current status is just edss_19)
def get_train_df(label_type):
    '''
    Get train neurology notes data (preprocessed by de-id script)
    Inputs: 
        label_type: type of label (edss_19, etc)
    Outputs: 
        text_df: dataframe of notes
        y_true: true labels of label_type
    '''

    path = "master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    df = df.fillna(-1)
    df = df[df.edss_19 != -1]
    text_df = df['text'].to_frame()

    # convert edss_19 to categorical from 0 - 18
    df['edss_19'] = convert_edss_19_to_categorical(df)
    # convert score 19 to -1
    df['edss_19'] = df['edss_19'].replace(19, -1)

    if label_type != "edss_19":
        text_df["edss_19"] = df['edss_19'].values

    y_true = df[label_type].values
    
    return text_df, y_true, path

def get_valid_df(label_type):
    '''
    Get valid neurology notes data (preprocessed by de-id script)
    Inputs: 
        label_type: type of label (edss_19, etc)
    Outputs: 
        text_df: dataframe of notes
        y_true: true labels of label_type
    '''
    path = "master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    df = df.fillna(-1)
    df = df[df.edss_19 != -1]
    text_df = df['text'].to_frame()

    # convert edss_19 to categorical from 0 - 18
    df['edss_19'] = convert_edss_19_to_categorical(df)
    # convert score 19 to -1
    df['edss_19'] = df['edss_19'].replace(19, -1)
    
    if label_type != "edss_19":
        text_df["edss_19"] = df['edss_19'].values

    y_true = df[label_type].values
    
    return text_df, y_true, path

def get_test_df(label_type):
    '''
    Get valid neurology notes data (preprocessed by de-id script)
    Inputs: 
        label_type: type of label (edss_19, etc)
    Outputs: 
        text_df: dataframe of notes
        y_true: true labels of label_type
    '''
    path = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
    df = pd.read_csv(path)
    df = df.dropna(subset = ['text'])
    # Fill NA values in labels to -1
    df = df.fillna(-1)
    df = df[df.edss_19 != -1]
    text_df = df['text'].to_frame()

    # convert edss_19 to categorical from 0 - 18
    df['edss_19'] = convert_edss_19_to_categorical(df)
    # convert score 19 to -1
    df['edss_19'] = df['edss_19'].replace(19, -1)
    
    if label_type != "edss_19":
        text_df["edss_19"] = df['edss_19'].values
        
    y_true = df[label_type].values

    return text_df, y_true, path
