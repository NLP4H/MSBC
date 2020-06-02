import pandas as pd
import numpy as np
import os
from os import path
import sys
import pdb

# helper function 
def convert_edss_19_to_categorical(labels_df, col_name):
    '''
    Convert edss_19 score from dataset into 19 categorical variables
    Inputs: df: dataframe of train, test or raw data
    '''
    edss_original = labels_df[col_name].values
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

# merge results
def merge_results( final_csv_path, incoming_result_path, prediction_col_name):
    """
    input:
        - final csv path which contains patient info, true label, and predictions made
        - incoming results path which contains new predictions to add to final results
        - prediction col name - name of prediction column to add
    output:
    save file: adds new prediction column to final results
    """
    ##
    save_dir = "master_path/results/task1/"
    output_name = "edss_19_results.csv"
    ##
    pdb.set_trace()
    # read in results
    print("reading in data")
    df_final = pd.read_csv(final_csv_path)
    df_pred = pd.read_csv(incoming_result_path)

    # make sure date is in correct format
    print("formatting data for merge")
    # uncomment if necessary
    # df_pred = df_pred.rename(columns={'date':'visit_date'})
    ###

    # uncomment if necessary
    del df_final['Unnamed: 0']
    del df_pred['Unnamed: 0']
    ###

    df_final.visit_date = pd.to_datetime(df_final.visit_date)
    df_pred.visit_date = pd.to_datetime(df_pred.visit_date)

    # sort and add index for merging 
    # df_final = df_final.sort_values(['patient_id','visit_date'])
    df_pred = df_pred.sort_values(['patient_id', 'visit_date'])

    # df_final = df_final.reset_index(drop=True)
    # df_final = df_final.reset_index()
    # print(df_final.head())
    df_pred = df_pred.reset_index(drop=True)
    df_pred = df_pred.reset_index()
    print(df_pred.head())

    print("merging")
    df_final=pd.merge(df_final,df_pred[['index','patient_id','visit_date',prediction_col_name]], how = 'inner', on=['index','patient_id','visit_date'])
    
    # print("converting edss_19 and predictions to categorical")
    # df_final['edss_19'] = convert_edss_19_to_categorical(df_final, 'edss_19')
    # df_final[prediction_col_name] = convert_edss_19_to_categorical(df_final, prediction_col_name)

    print(df_final.head())
    print(df_final.shape)
    print('Saving result')

    os.chdir(save_dir)
    df_final.to_csv(output_name)
    print("done :)")

    return


#uncomment below to run
# final_path = "master_path/results/task1/edss_19_results.csv"
# incoming_path = "master_path/results/task1/predict_edss_19_snorkel_rb_tfidf_allen_cnn_lm.csv"
# prediction_name = 'predict_edss_19_snorkel_rb_tfidf_allen_cnn_lm'

# merge_results(final_path,incoming_path,prediction_name)
