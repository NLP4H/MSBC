import pandas as pd
import numpy as np
import re
import os
import sys
import pdb

# labelling functions
from edss_19_lfs import get_snorkel_compatible_lfs
sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning/')
import snorkel_functions
from snorkel_functions import get_num_classes
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel


def prediction(labels, eval_type):
    '''
    Obtain 1 label prediction per data point
    Input: 
        labels: labels for each data point for every lf
        eval_type: type of model used to obtain single prediction
    Output: 
        y_preds: single label prediction for each data point, np array
    '''
    label_type = 'edss_19'
    num_classes = get_num_classes(label_type)

    if eval_type == "majority_vote":
        # majority vote model to obtain predictions
        majority_model = MajorityLabelVoter(cardinality=num_classes)
        y_preds = majority_model.predict(L=labels)
    
    elif eval_type == "label_model":
        # label model to obtain predictions
        label_model = LabelModel(cardinality=num_classes, verbose=True)      # cardinality = number of classes
        label_model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=123)
        y_preds = label_model.predict(L=labels)
    
    return y_preds

def predict_snorkel(df_result_path, prediction_col_names, eval_type, model_type):   
    """
    input: df path of results, prediction column names to use for snorkel input, eval_type, model type (name)
    output: saved .csv
    saves .csv in format: patient_id|date|split|text|edss_19|predicted_edss_19_<model_type>
    """
    print("reading in df")
    # drop nans in text and labels (i.e. -1)
    df = pd.read_csv(df_result_path)
    empty_text = df[df.text.isna()]
    df = df.dropna(subset = ['text'])

    df_unlabeled = df[df.split=='unlabeled']
    df = df[df.split!='unlabeled']

    df.edss_19 = df.edss_19.fillna(-1)
    empty_labels = df[df.edss_19==-1]
    df = df[df.edss_19 != -1]

    # format [[label_1,label_2,]]
    labels = df[prediction_col_names].to_numpy()
    labels_pred = df_unlabeled[prediction_col_names].to_numpy()
    
    # # Save rules
    # #rb labels
    # lfs_snorkel_compatible = get_snorkel_compatible_lfs()
    # applier = PandasLFApplier(lfs=lfs_snorkel_compatible)
    # labels_rb = applier.apply(df=df)
    # labels_rb_unlabeled = applier.apply(df=df_unlabeled)
    save_dir = "master_path//repo/ML4H_MSProject/data/baseline_data/"
    os.chdir(save_dir)
    # np.savez("snorkel_rb_labels.npz", labels_rb, labels_rb_unlabeled)

    # load rules
    npzfile = np.load(save_dir +"snorkel_rb_labels.npz")
    labels_rb = npzfile['arr_0'] 
    labels_rb_unlabeled = npzfile['arr_1'] #test set

    os.chdir("master_path/results/task1/")
    labels = np.column_stack((labels,labels_rb))
    labels_pred = np.column_stack((labels_pred,labels_rb_unlabeled))
  
    # fill prediction for empty text entries with -1
    empty_text['predict_edss_19_' + model_type] = -1
    empty_labels['predict_edss_19_'  + model_type] = -1


    #predict and save file
    print("predicting")
    print("model_type: " + model_type)
    predict = prediction(labels, eval_type)
    df['predict_edss_19_' + model_type] = predict
    predict_unlabeled = prediction(labels_pred, eval_type)
    df_unlabeled['predict_edss_19_' + model_type] = predict_unlabeled


    # keep only columns we are concerned with
    print("keeping only columns we are concerned with")
    df = df[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    df_unlabeled = df_unlabeled[['patient_id','visit_date', 'gender', 'split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_text = empty_text[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    empty_labels = empty_labels[['patient_id','visit_date','gender','split', 'text', 'edss_19', 'predict_edss_19_'+model_type]]
    #concatenate all dataframes
    print("concat")
    df_result = pd.concat([df, df_unlabeled, empty_text, empty_labels], ignore_index=True)
    # pdb.set_trace()

    # save final dataframe result
    print("saving")
    save_dir = "master_path/results/task1/"
    output_name = "predict_edss_19_"+model_type+".csv"
    os.chdir(save_dir)
    df_result.to_csv(output_name)
    print("shape of final dataframe")
    print(df_result.shape)
    print("done :)")
    
    return


################### predictioons we can mix and match with for snorkel labeling generation
# predict_edss_19_word2vec_cnn
# predicted_edss_19_allen_cnn
# predict_edss_19_log_reg_baseline
# predict_edss_19_linear_svc
# predict_edss_19_svc_rbf
# predict_edss_19_lda
###################

# # uncomment below to run
# #rb always ran with additional prediction models below
# result_path = "master_path/results/task1/edss_19_results.csv"
# prediction_col = ['predicted_edss_19_allen_cnn'] #, 'predict_edss_19_lda', 'predict_edss_19_log_reg_baseline', 'predict_edss_19_linear_svc', 'predict_edss_19_svc_rbf'
# model_t = "snorkel_rb_allen_cnn_lm" #snorkel_rb_allen_cnn_mv
# eval_t = "label_model" #"majority_vote","label_model"
# predict_snorkel(result_path,prediction_col,eval_t,model_t)   