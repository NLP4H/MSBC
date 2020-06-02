# Helper functions to preprocess the note, tokenize the note, text_to_sequence
import pandas as pd
import numpy as np
import os
from os import path
import sys
# will need to use GPU if training model file - uncomment line below
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import re
import tensorflow as tf
import string
#keras
from keras.preprocessing.text import Tokenizer
#gensim
import pickle
from collections import defaultdict
#nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#sklearn
from sklearn.utils import class_weight
# Sklearn models imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
#helper functions
sys.path.append('master_path//repo/ML4H_MSProject/semi-supervised-learning')
import load_data
from load_data import get_raw_df, get_train_df, get_valid_df, get_test_df

import pdb

###########
## HELPER FUNCTIONS
###########

# preprocessed neurology note (removes stop words)
def tokenize_neurology(note):
    """
    Pre-processing for neurology notes
    Input: one piece of neurology note
    Output: 
        pre-processed piece of neurology note
        filtered: a list version of each word in the note ["word1","word2", ...] used for word2vec creation
    """ 
    # -----------------Text Level Preproc--------------
    note = note.lower()
    # Stop words
    stop_words = set(stopwords.words('english'))
    # Punctuations except '.'
    punc = string.punctuation.replace(".", "")
    translator = str.maketrans('', '', punc)
    # Remove irrelevant punctuations
    note = note.translate(translator)
    # Tokenization
    tokens = word_tokenize(note) 
    # Remove stop words
    filtered = [w for w in tokens if not w in stop_words]
    # note = " ".join(filtered)
    return filtered

# train tfidf models
def train_models(models_to_train):
    """
    train models and hyperparam searches
    input:
        models to train as list of string (i.e. ["logreg","mnb","cnb", etc...])
    output:
    save: models
    """
    # load train and val data
    label_type = "edss_19"
    df_train, y_train_true, _ = load_data.get_train_df(label_type)
    df_valid, y_valid_true, _ = load_data.get_valid_df(label_type)

    # load class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_true), y_train_true)
    class_weights_dict = {}
    for i in range(19):
        class_weights_dict[i] = class_weights[i]

    # load tfidf vectorizer
    tf = TfidfVectorizer(tokenizer=tokenize_neurology, max_features=1500)
    
    #get notes into list
    texts_train = list(df_train['text'])
    texts_valid = list(df_valid['text'])
    #make type numpy
    texts_train = np.array(texts_train)
    texts_valid = np.array(texts_valid)

    # fit tfidfvectorizer to train and valid
    X_train = tf.fit_transform(texts_train)
    
    # save_dir = "master_path//repo/ML4H_MSProject/data/baseline_models/"
    # os.chdir(save_dir)
    # model_name = "tf.pkl"
    # with open(model_name, 'wb') as file:
    #     pickle.dump(tf, file)
    
    X_valid = tf.transform(texts_valid)

    # Hyper Param Tuning:
    tuned_params_LR = [{"C": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
        1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]

    tuned_params_MNB = [{"alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
        1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]

    tuned_params_CNB = [{"alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
        1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]

    tuned_params_SVC = [{"C": [0.01, 0.1, 1.0, 10.0, 100.0]}]

    save_dir = "master_path//repo/ML4H_MSProject/data/baseline_models/"

    for clf_name in models_to_train:
        #logreg classifier
        if clf_name == "log_reg_baseline":
            print("hyperparam tuning for LR")
            best_c = 0
            best_score = 0
            for i, param in enumerate(tuned_params_LR[0]["C"]):
                clf = LogisticRegression(C=param, 
                    class_weight=class_weights_dict,solver='lbfgs')
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of logistic regression model with C:",param)
                print(score)
                if float(score) > float(best_score):
                    best_score = score
                    best_c = param
            #save best model
            print("training model to best:", best_c)
            clf = LogisticRegression(C=best_c, 
                class_weight=class_weights_dict,solver='lbfgs')
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "log_reg_baseline.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
        # mnb classifier
        elif clf_name == "multinomial_nb":
            print("hyperparam tuning for MNB")
            best_alpha = 0
            best_score = 0
            for i, param in enumerate(tuned_params_MNB[0]["alpha"]):
                clf = MultinomialNB(alpha=param)
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of multinomial_nb model with alpha:",param)
                print(score)
                if float(score) > float(best_score):
                    best_score = score
                    best_alpha = param
            #save best model
            print("training model to best:", best_alpha)
            clf = MultinomialNB(alpha=best_alpha)
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "multinomial_nb.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
        # cnb classifier
        elif clf_name == "complement_nb":
            print("hyperparam tuning for CNB")
            best_alpha = 0
            best_score = 0
            for i, param in enumerate(tuned_params_CNB[0]["alpha"]):
                clf = ComplementNB(alpha=param)
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of complement_nb model with alpha:",param)
                print(score)
                if score>best_score:
                    best_score = score
                    best_alpha = param
            #save best model
            print("training model to best:", best_alpha)
            clf = ComplementNB(alpha=best_alpha)
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "complement_nb.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
        # training LDA
        elif clf_name == "lda":
            print("Creating LDA")
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train.todense(),y_train_true)
            score = accuracy_score(clf.predict(X_valid.todense()),y_valid_true)
            print("score of lda model:",score)
            #save
            os.chdir(save_dir)
            model_name = "lda.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)  
        # training svc_rbf
        elif clf_name == "svc_rbf":
            print("hyperparam tuning for svc_rbf")
            best_c = 0
            best_score = 0
            for i, param in enumerate(tuned_params_SVC[0]["C"]):
                clf = SVC(kernel = "rbf", C = param, class_weight = class_weights_dict)
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of svc_rbf model with C:",param)
                print(score)
                if float(score) > float(best_score):
                    best_score = score
                    best_c = param
            #save best model
            print("training model to best:", best_c)
            clf = SVC(kernel = "rbf", C = best_c, class_weight = class_weights_dict)
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "svc_rbf.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
        # create svc_sigmoid
        elif clf_name == "svc_sigmoid":
            print("hyperparam tuning for svc_sigmoid")
            best_c = 0
            best_score = 0
            for i, param in enumerate(tuned_params_SVC[0]["C"]):
                clf = SVC(kernel = "sigmoid", C = param, class_weight = class_weights_dict)
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of svc_sigmoid model with C:",param)
                print(score)
                if float(score) > float(best_score):
                    best_score = score
                    best_c = param
            #save best model
            print("training model to best:", best_c)
            clf = SVC(kernel = "sigmoid", C = best_c, class_weight = class_weights_dict)
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "svc_sigmoid.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)
        #training linear_svc
        elif clf_name == "linear_svc":
            print("hyperparam tuning for linear_svc")
            best_c = 0
            best_score = 0
            for i, param in enumerate(tuned_params_SVC[0]["C"]):
                clf = LinearSVC(penalty = "l1", dual=False, C = param, class_weight = class_weights_dict)
                clf.fit(X_train,y_train_true)
                score = accuracy_score(clf.predict(X_valid),y_valid_true)
                print("score of linear_svc model with C:",param)
                print(score)
                if float(score) > float(best_score):
                    best_score = score
                    best_c = param
            #save best model
            print("training model to best:", best_c)
            clf = LinearSVC(penalty = "l1", dual = False, C = best_c, class_weight = class_weights_dict)
            clf.fit(X_train,y_train_true)
            os.chdir(save_dir)
            model_name = "linear_svc.pkl"
            with open(model_name, 'wb') as file:
                pickle.dump(clf, file)                                        
        else:
            print("model yet to be implemented")
    return

##########
# TRAIN MODELS
##########
# FOUND MNB, CNB, SVC_SIGMOID TO PERFORM POORLY (<70% ACC)
# train_models([]) # "log_reg_baseline", "multinomial_nb", "complement_nb", "lda", "svc_rbf", "svc_sigmoid", "linear_svc" 
