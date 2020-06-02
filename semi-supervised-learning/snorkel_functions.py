# SNORKEL FUNCTIONS
# Contains functions needed to label notes data using snorkel

import pandas as pd
import numpy as np
import load_data
from load_data import get_raw_df, get_train_df, get_valid_df, get_test_df
import snorkel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import MajorityLabelVoter
from snorkel.labeling import LabelModel
from snorkel.labeling import LFAnalysis
from snorkel.analysis import get_label_buckets
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import sys

sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning/edss_19')
from edss_19_lfs import LF_edss_cnn_word2vec, LF_edss_tfidf_logreg,LF_edss_tfidf_lda, LF_edss_tfidf_svc_rbf, LF_edss_tfidf_linear_svc
from edss_19_lfs import get_snorkel_compatible_lfs


def get_num_classes(label_type):
    '''
    Gets number of classes for each type of label
    Input: label type
    Output: number of classes for that label
    '''

     # TODO check, is cerebral = mental?
    num_classes_dict = {'edss_19':19,
        'score_brain_stem_subscore':7,
        'score_pyramidal_subscore':8,
        'score_cerebellar_subscore':8,
        'score_sensory_subscore':8,
        'score_bowel_bladder_subscore':8,
        'score_ambulation_subscore':16,
        'score_visual_subscore':8,
        'score_mental_subscore':7}
    return num_classes_dict[label_type]

def get_label_predictions(df, lfs, df_path):
    '''
    Each labelling function predicts a label per datapoint using PandasLFApplier
    Input: 
        df: data containing notes
        lfs: labelling functions for current label type
    Output: 
        labels for each data point per each labelling function
    '''
    # some keras models are incompatible with snorkel=0.93 -> predicting outside of snorkel and combining labels.    
    # TODO use this when you need to test without cnn
    #lfs_snorkel_compatible = lfs
    #applier = PandasLFApplier(lfs=lfs_snorkel_compatible)
    #labels = applier.apply(df=df)
    lfs_snorkel_compatible = get_snorkel_compatible_lfs()
    applier = PandasLFApplier(lfs=lfs_snorkel_compatible)
    labels = applier.apply(df=df)
    #predict offline snorkel incompatible modules
    for i in lfs:
        if i == LF_edss_cnn_word2vec:
            labels_cnn = LF_edss_cnn_word2vec(df_path)
            #append offline predictions to snorkel labels
            labels = np.column_stack(( labels, labels_cnn))
        elif i == LF_edss_tfidf_logreg:
            labels_tfidf_logreg = LF_edss_tfidf_logreg(df_path)
            #append offline predictions to snorkel labels
            labels = np.column_stack(( labels, labels_tfidf_logreg))
        elif i == LF_edss_tfidf_lda:
            labels_tfidf_lda = LF_edss_tfidf_lda(df_path)
            #append offline predictions to snorkel labels
            labels = np.column_stack(( labels, labels_tfidf_lda))
        elif i == LF_edss_tfidf_svc_rbf:
            labels_tfidf_svc_rbf = LF_edss_tfidf_svc_rbf(df_path)
            #append offline predictions to snorkel labels
            labels = np.column_stack(( labels, labels_tfidf_svc_rbf))
        elif i == LF_edss_tfidf_linear_svc:
            labels_tfidf_linear_svc = LF_edss_tfidf_linear_svc(df_path)
            #append offline predictions to snorkel labels
            labels = np.column_stack(( labels, labels_tfidf_linear_svc))
        else:
            pass
    return labels

def label_data(lfs, label_type, data_type):
    '''
    Load train, valid or raw data and label them 
    Input: 
        lfs: labelling functions for current label type
        label_type : type of label
        data_type: train, valid or raw
    Output: 
        labels for each data point for every lf
    '''

    if data_type == "train":
        text_df, y_true, df_path = load_data.get_train_df(label_type)
        # Label data
        labels = get_label_predictions(text_df, lfs, df_path)
        
        return labels, y_true

    elif data_type == "valid":
        text_df, y_true, df_path = load_data.get_valid_df(label_type)
        # Label data
        labels = get_label_predictions(text_df, lfs, df_path)
        
        return labels, y_true

    elif data_type == "test":
        text_df, y_true, df_path = load_data.get_test_df(label_type)
        # Label data
        labels = get_label_predictions(text_df, lfs, df_path)
        
        return labels, y_true

    elif data_type == "raw":
        text_df, df_path = load_data.get_raw_df()
        labels = get_label_predictions(text_df, lfs, df_path)

        return labels

def get_predictions(labels, lfs, label_type, model_type):
    '''
    Obtain 1 label prediction per data point
    Input: 
        labels: labels for each data point for every lf
        lfs: labelling functions for current label type
        label_type : type of label
        model_type: type of model used to obtain single prediction
    Output: 
        y_preds: single label prediction for each data point, np array
    '''
    num_classes = get_num_classes(label_type)

    if model_type == "majority_vote":
        # majority vote model to obtain predictions
        majority_model = MajorityLabelVoter(cardinality=num_classes)
        y_preds = majority_model.predict(L=labels)
    
    elif model_type == "label_model":
        # label model to obtain predictions
        label_model = LabelModel(cardinality=num_classes, verbose=True)      # cardinality = number of classes
        label_model.fit(L_train=labels, n_epochs=500, log_freq=100, seed=123)
        y_preds = label_model.predict(L=labels)
    
    return y_preds

def evaluate_predictions(labels, y_true, lfs, label_type):
    '''
    Evaluate prediction accuracy for each prediction model and return the predictions with the highest accuracy
    Input: 
        labels: labels for each data point for every lf
        y_true: true labels for each datapoint
        lfs: labelling functions for current label type
        label_type : type of label
    Output: 
        y_preds with highest accuracy 
    '''
    # If there is only 1 labelling function --> no need to vote
    if len(lfs) == 1:
        y_preds = labels.reshape(len(labels))
        acc = calculate_prediction_accuracy(y_preds, y_true, label_type)
        print("Accuracy = " + str(acc))
        return y_preds

    # If there are more than 1 labelling functions
    else:
        # majority vote model to obtain predictions
        print("Accuracy of data labelling with majority vote model: ")
        y_preds_majority = get_predictions(labels, lfs, label_type, "majority_vote")
        majority_acc = calculate_prediction_accuracy(y_preds_majority, y_true, label_type)
        print("Accuracy = " + str(majority_acc))

        # label model to obtain predictions
        print("Accuracy of data labelling with label model: ")
        y_preds_label = get_predictions(labels, lfs, label_type, "label_model")
        label_acc = calculate_prediction_accuracy(y_preds_label, y_true, label_type)
        print("Accuracy = " + str(label_acc))

        if majority_acc > label_acc:
            print("More accurate prediction model = MAJORITY VOTE MODEL, accuracy = " + str(majority_acc))
            return y_preds_majority
        else:
            print("More accurate prediction model = SNORKEL LABEL MODEL, accuracy = " + str(label_acc))
            return y_preds_label

def calculate_prediction_accuracy(y_preds, y_true, label_type):
    '''
    Calculate prediction accuracy 
    Input: 
        y_preds: predicted labels for each data point
        y_true: true labels for each data point
        lfs: labelling functions for current label type
        label_type : type of label
    Output: 
        accuracy
        edss_19 = calculate exact accuracy
        subscores = calculate accuracy to +-1
    '''
    print(classification_report(y_true, y_preds))

    # EDSS calculates exact accuracy
    if label_type == 'edss_19':
        return accuracy_score(y_true, y_preds)
    
    # Functional subscore calculates accuracy within +- 1
    else:
        total_count = 0 
        count = 0
        for i in range(len(y_preds)):
           # If predicted label is not unknown
            if y_preds[i] != -1 and y_true[i] != -1:
                total_count += 1
                # If difference between label and prediction is less than or equal to 1
                if abs(y_preds[i] - y_true[i]) <= 1:
                    count += 1
        return count / total_count

def rb_cnn_exploratory(labels, y_true):
    y_pred=[]
    for i in labels:
        rb = i[0:3]
        if np.all(rb==-1):
            y_pred.append(i[4])
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
    print(classification_report(y_true, y_pred, digits=4))
    return(accuracy_score(y_true,y_pred))
        
def calculate_coverage(y_preds, y_true):
    num_present = 0
    num_predicted = 0
    for i in range(len(y_true)):
        if y_true[i] != -1:
            num_present += 1
            
            # if predicted value known WHEN true value is also known
            if y_preds [i] != -1:
                num_predicted += 1

    print("% Coverage (Number predicted when true label is known/ all known true labels) : " + str(100 * (num_predicted / num_present)))

def review_incorrectly_labelled_texts(y_preds, data_type, label_type):
    '''
    Review data that was incorrectly labelled in train / valid data 
    Inputs: 
        y_preds: predicted labels from snorkel lfs
        data_type: train, valid
    Outputs:
        incorrect_df: dataframe of notes that were inaccurately labelled 
    '''
    if data_type == "train":
        df, y_true, _ = load_data.get_train_df(label_type)

    elif data_type == "valid":
        df, y_true, _ = load_data.get_valid_df(label_type)
    
    elif data_type == "test":
        df, y_true, _ = load_data.get_test_df(label_type)

    df['y_true'] = y_true.tolist()
    df['y_preds'] = y_preds.tolist()

    if label_type == "edss_19":
        # exclude rows where true label is unknown
        incorrect_df = df.loc[(df['y_true'] != -1)]
        # get rows where prediction is not exactly accurate
        incorrect_df = incorrect_df.loc[(incorrect_df['y_true'] != incorrect_df['y_preds'])]
    else:
        # exclude rows where true label is unknown
        incorrect_df = df.loc[(df['y_true'] != -1)]
        # get rows where prediction is not accurate with +- 1 tolerance
        incorrect_df = incorrect_df.loc[abs(incorrect_df['y_true'] - incorrect_df['y_preds']) >=2]
    return incorrect_df