# Evaluation of combined model from both CNN and rules
# Output: classification report, confusion matrix, ROC
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
print(tf.__version__)
import sys
import importlib
from keras.models import load_model
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import rules
from allennlp.common.params import Params
# Sklearn models imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import pdb

def prediction_rules(notes, variable_name, model_dir, edss = []):
    
    """
    Batch predictions with rule-based model
    
    Input:
    notes: list of notes
    variable_name: variable to predict
    model_dir: directory of saved rule-based models

    Return:
    List of predictions by rules

    """
    # import rules file
    sys.path.insert(1, model_dir)
    mod = importlib.import_module(variable_name)
    raw_preds = []

    # Make predictions
    # Predictions needs EDSS
    if variable_name in ["score_brain_stem_subscore", 
                         "score_cerebellar_subscore", 
                         "score_pyramidal_subscore", 
                         "score_sensory_subscore", 
                         "score_bowel_bladder_subscore", 
                         "score_ambulation_subscore",
                         "score_mental_subscore",
                         "score_visual_subscore"]:
        for i in range(len(notes)):
            raw_preds.append(mod.predict(edss[i], notes[i]))
    # Predictions don't need EDSS
    else:
        for i in range(len(notes)):
            raw_preds.append(mod.predict(notes[i]))
    
    return raw_preds

def evaluate_rb_cnn(var, test_dir, baseline_data, baseline_labels, model_path, baseline_code_path, with_rb):
    
    # ------------------- Load Data and Model -----------------------------------
    # Variable to predict
    var = var

    print("Variable name: ", var)
    print("Loading validation data ...")   
    # Load notes
    val_dir = test_dir
    note_column = "text"

    df_val = pd.read_csv(val_dir)
    print(df_val.shape)
    # Remove places where there's no note
    df_val = df_val.dropna(subset = [note_column])
    # Fill NA values in labels to -1
    df_val = df_val.fillna(-1)
    # Reset Index
    df_val = df_val.reset_index()
    notes = df_val[note_column].tolist()
    print(df_val.shape)
    # Load X_val

    print("Loading Data ...")

    npzfile = np.load(baseline_data +"training_data_neurology.npz")
    x_val = npzfile['arr_2']
    print(x_val.shape)
    # Load y_val
    npzfile = np.load(baseline_labels + "edss_19.npz")
    y_val = npzfile['arr_2']
    print(y_val.shape)
    # Load Model
    print("Loading models ...")
    model = load_model(model_path +"edss_19.h5") # TODO: Directory that saves the .h5 model files


    # ------------------- Rule-based prediction -------------------------
    print("Prediction variable: ", var)
    print("Rule-based predictions ...")
    raw_preds = prediction_rules(notes, var, baseline_code_path)

    # ------------------- CNN based prediction -------------------------
    print("CNN predictions ...")
    #print(model.get_config())
    y_pred = model.predict(x_val)
    print("predictions complete")
    y_pred_class = y_pred.argmax(axis = -1)
    y_val_class = y_val.argmax(axis = -1)

    # --------------------- Combined prediction -----------------------------
    # Get combined model prediction
    print("Combining models ... ")
    final_preds = []
    if(with_rb):
        print("perfomring with rb")
        for i in range(len(raw_preds)):
            if raw_preds[i] != -1:
                final_preds.append(raw_preds[i])
            if raw_preds[i] == -1:
                final_preds.append(y_pred_class[i])
    else:
        print("performing without rb")
        for i in range(len(y_pred_class)):
            final_preds.append(y_pred_class[i])
    

    # -------------------- Classification report --------------------------
    # Print classification report
    print("Classification Report: ")
    print(classification_report(y_val_class, final_preds, digits=4))

    # --------------- Converted accuracy -------------
    print("Converted Accuracy (if score is between label +-1)")
    unknown_class = sorted(set(y_val_class))[-1] # Define the class for Unknown prediction
    total_count = 0
    count = 0
    for i in range(len(final_preds)):
        # If both label and prediction is not Unknown
        if y_val_class[i] != unknown_class and final_preds[i] != unknown_class:
            total_count += 1
            # If difference between label and prediction is less than or equal to 1
            if abs(y_val_class[i] - final_preds[i]) <= 1:
                count += 1
    print(count / total_count)

def evaluate_rb_countvectorizer_tfidf(var, config_file_path, baseline_labels_path, baseline_code_path, note_column = "text", count_vec = False, tf = True, with_rb = False):
    
    # ------------------- Load Data and Model -----------------------------------

    print("Variable name: ", var)
    print("Loading validation data ...")
    config = Params.from_file(config_file_path)

    params = config["baselines"]
	# Read inputs
    df_train = pd.read_csv(params["data_dir_train"])
    df_valid = pd.read_csv(params["data_dir_valid"])

    print(df_train.shape)
    print(df_valid.shape)

    # Remove places where there's no note
    df_train = df_train.dropna(subset = [note_column])
    # Fill NA values to -1
    df_train = df_train.fillna(-1)
    # Reset Index
    df_train = df_train.reset_index()
    # Remove places where there's no note
    df_valid = df_valid.dropna(subset = [note_column])
    # Fill NA values to -1
    df_valid = df_valid.fillna(-1)
    # Reset Index
    df_valid = df_valid.reset_index()
    notes = df_valid[note_column].tolist()

    os.chdir(baseline_labels_path)
    # Load y_train, y_val
    npzfile = np.load(var + ".npz")
    Y_train = npzfile['arr_0'] 
    Y_valid = npzfile['arr_2'] #test set

    # Fill in the count vectorizer
    X_train = df_train.text.to_numpy(copy=True)
    X_valid = df_valid.text.to_numpy(copy=True)
    print(X_train.shape)
    print(Y_train.shape)

    print(X_valid.shape)
    print(Y_valid.shape)

    if(count_vec==True):
        tf1 = CountVectorizer()
    else:
        tf1 = TfidfVectorizer()

    X_train = tf1.fit_transform(X_train)
    X_valid = tf1.transform(X_valid)

    print(X_train.shape)
    print(Y_train.shape)

    # Load Models
    print("Loading models ...")
    log_reg_baseline = MultiOutputClassifier(
        LogisticRegression(**params["logreg"]))

    multinomial_nb = MultiOutputClassifier(
        MultinomialNB(**params["multinomial_nb"]))

    complement_nb = MultiOutputClassifier(
        ComplementNB(**params["complement_nb"]))

    svm_rbf = MultiOutputClassifier(
        SVC(**params["svm_rbf"]))

    # svm_polynomial = MultiOutputClassifier(
    # SVC(**params["svm_polynomial"]))

    svm_sigmoid = MultiOutputClassifier(
        SVC(**params["svm_sigmoid"]))

    linear_svc = MultiOutputClassifier(
        LinearSVC(**params["linear_svc"]))

    lda = MultiOutputClassifier(
        LinearDiscriminantAnalysis())

    qda = MultiOutputClassifier(
        QuadraticDiscriminantAnalysis(**params["qda"]))

    baseline_classifiers = {
        "log_reg_baseline": log_reg_baseline,
        "multinomial_nb": multinomial_nb,
        "complement_nb": complement_nb,
        "svc_rbf": svm_rbf,
        "svc_sigmoid": svm_sigmoid,
        "linear_svc": linear_svc,
        "lda": lda
        # "qda": qda
    }

    # ------------------- Rule-based prediction -------------------------
    if(with_rb):
        print("Prediction variable: ", var)
        print("Rule-based predictions ...")
        raw_preds = prediction_rules(notes, var, baseline_code_path)

    # ------------------- Model based prediction -------------------------
    print("Model predictions ...")
    # FIT
    ######
    #really hacky way to fix the fit when we have no class edss 18 examples
    # pdb.set_trace()
    Y_valid[0]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    Y_train[0]=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
    ######
    y_val_class = Y_valid.argmax(axis = -1)
    for clf_name, clf in baseline_classifiers.items():
        print("Evaluating model: %s" %(clf_name))
        if "svc" in clf_name:
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_valid)
        elif "qda" or "lda" in clf_name:
            clf.fit(X_train.todense(), Y_train)
            y_pred = clf.predict(X_valid.todense())
        else:
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_valid)
        print("predictions complete")
        y_pred_class = y_pred.argmax(axis = -1)
        # --------------------- Combined prediction -----------------------------
        if(with_rb):
            # Get combined model prediction
            print("Combining models ... ")
            final_preds = []
            for j in range(len(raw_preds)):
                if raw_preds[j] != -1:
                    final_preds.append(raw_preds[j])
                if raw_preds[j] == -1:
                    final_preds.append(y_pred_class[j])
        else:
            final_preds = []
            for j in range(len(y_pred_class)):
                final_preds.append(y_pred_class[j])

        # -------------------- Classification report --------------------------
        # Print classification report
        print("Classification Report: ")
        print(classification_report(y_val_class, final_preds, digits=4))

        # --------------- Converted accuracy -------------
        print("Converted Accuracy (if score is between label +-1)")
        unknown_class = sorted(set(y_val_class))[-1] # Define the class for Unknown prediction
        total_count = 0
        count = 0
        for j in range(len(final_preds)):
            # If both label and prediction is not Unknown
            if y_val_class[j] != unknown_class and final_preds[j] != unknown_class:
                total_count += 1
                # If difference between label and prediction is less than or equal to 1
                if abs(y_val_class[j] - final_preds[j]) <= 1:
                    count += 1
        print(count / total_count)
    return

def evaluate_rb(data_dir_test, var, baseline_labels_path, baseline_code_path, note_column="text" ):
    # ------------------- Load Data and Model -----------------------------------

    print("Variable name: ", var)
    print("Loading validation data ...")

	# Read inputs
    df_valid = pd.read_csv(data_dir_test)

    # Remove places where there's no note
    df_valid = df_valid.dropna(subset = [note_column])
    # Fill NA values to -1
    df_valid = df_valid.fillna(-1)
    # Reset Index
    df_valid = df_valid.reset_index()
    notes = df_valid[note_column].tolist()

    os.chdir(baseline_labels_path)
    # Load y_train, y_val
    npzfile = np.load(var + ".npz")
    Y_valid = npzfile['arr_2'] 

    # Fill in the count vectorizer
    X_valid = df_valid.text.to_numpy(copy=True)

    print("Prediction variable: ", var)
    print("Rule-based predictions ...")
    raw_preds = prediction_rules(notes, var, baseline_code_path)

    y_val_class = Y_valid.argmax(axis = -1)
    # -------------------- Classification report --------------------------
    # Print classification report
    print("Classification Report: ")
    print(classification_report(y_val_class, raw_preds, digits=4))

    # --------------- Converted accuracy -------------
    print("Converted Accuracy (if score is between label +-1)")
    unknown_class = sorted(set(y_val_class))[-1] # Define the class for Unknown prediction
    total_count = 0
    count = 0
    for j in range(len(raw_preds)):
        # If both label and prediction is not Unknown
        if y_val_class[j] != unknown_class and raw_preds[j] != unknown_class:
            total_count += 1
            # If difference between label and prediction is less than or equal to 1
            if abs(y_val_class[j] - raw_preds[j]) <= 1:
                count += 1
    print(count / total_count)
    return

# config_path = "master_path//repo/ML4H_MSProject/task1/configs/baselines.jsonnet"
# var = "edss_19"
# data_dir_valid = "master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv"
# baseline_data = "master_path//repo/ML4H_MSProject/data/baseline_data/"
# baseline_labels = "master_path//repo/ML4H_MSProject/data/baseline_labels/"
# baseline_code_path = "master_path//repo/ML4H_MSProject/baseline/baseline_code/"
# model_path = "master_path//repo/ML4H_MSProject/data/baseline_models/"

# evaluate_rb_cnn(var,data_dir_valid,baseline_data, baseline_labels, model_path, baseline_code_path, with_rb=True)
# evaluate_rb(data_dir_valid, var, baseline_labels, baseline_code_path)
# evaluate_rb_countvectorizer_tfidf(var,config_path, baseline_labels, baseline_code_path, with_rb=True)