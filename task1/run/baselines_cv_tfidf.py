from allennlp.common.params import Params

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import logging
import random
import numpy as np
import pandas as pd

import pdb
# Sklearn models imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

# Hyperparam
from sklearn.model_selection import GridSearchCV



logger = logging.getLogger(__name__)
logging.basicConfig(level=20)


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)

def run_baselines(var, config_file_path, baseline_labels_path, note_column = "text", count_vec = False):
	config = Params.from_file(config_file_path)
	params = config["baselines"]

	# MAKE TFIDF HERE
	df_train = pd.read_csv(params["data_dir_train"])
	df_valid = pd.read_csv(params["data_dir_valid"])
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

	os.chdir(baseline_labels_path)
    # Load y_train, y_val
	npzfile = np.load(var + ".npz")
	Y_train = npzfile['arr_0'] 	
	Y_valid = npzfile['arr_2'] 

	# Fill in the  TFIDF
	X_train = df_train.text.to_numpy(copy=True)
	X_valid = df_valid.text.to_numpy(copy=True)
	print(X_train.shape)
	print(Y_train.shape)

	print(X_valid.shape)
	print(Y_valid.shape)

	if (count_vec):
		tf1 = CountVectorizer()
	else:
		tf1 = TfidfVectorizer()

	X_train = tf1.fit_transform(X_train)
	X_valid = tf1.transform(X_valid)

	print(X_train.shape)
	print(Y_train.shape)

	# pdb.set_trace()
	
	# Hyper Param Tuning:
	tuned_params_LR = [{"C": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
                           1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]
	tuned_params_LR_scores = []
	tuned_params_MNB = [{"alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
                                1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]
	tuned_params_MNB_scores = []
	tuned_params_CNB = [{"alpha": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5,
                                1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]
	tuned_params_CNB_scores = []
	tuned_params_QDA = [{"reg_param": [0.1, 0.25, 0.5, 0.75, 1.0,
                                    1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0, 10.0, 100.0]}]
	tuned_params_QDA_scores = []

	logger.info("Hyperparam Tuning for LR")
	max_score = 0
	max_score_param = 0
	for i, param in enumerate(tuned_params_LR[0]["C"]):
		clf = MultiOutputClassifier(LogisticRegression(C=param))
		# pdb.set_trace()
		clf.fit(X_train, Y_train)
		# scores = accuracy_score(clf.predict(X_valid), Y_valid)
		scores = classification_report(Y_valid, clf.predict(X_valid), digits=4,output_dict=True)
		score = scores['samples avg']["f1-score"]
		logger.info(f" Scores: {score} || Param Value: {param}")
		tuned_params_LR_scores.append(score)
		if score > max_score:
			max_score = score
			max_score_param = param

	logger.info(
		f"Best Parameter Value of {max_score_param} with a score of {max_score}")
	if max_score > 0:
		params["logreg"]["C"] = max_score_param

	logger.info("Hyperparam Tuning for MNB")
	max_score = 0
	max_score_param = 0
	for i, param in enumerate(tuned_params_MNB[0]["alpha"]):
		clf = MultiOutputClassifier(MultinomialNB(alpha=param))
		clf.fit(X_train, Y_train)
		scores =  classification_report(Y_valid, clf.predict(X_valid), digits=4, output_dict=True)
		score = scores['samples avg']["f1-score"]
		logger.info(f" Scores: {score} || Param Value: {param}")
		tuned_params_MNB_scores.append(score)
		if score > max_score:
			max_score = score
			max_score_param = param

	logger.info(
		f"Best Parameter Value of {max_score_param} with a score of {max_score}")
	if max_score > 0:
		params["multinomial_nb"]["alpha"] = max_score_param

	logger.info("Hyperparam Tuning for CNB")
	max_score = 0
	max_score_param = 0
	for i, param in enumerate(tuned_params_CNB[0]["alpha"]):
		clf = MultiOutputClassifier(ComplementNB(alpha=param))
		clf.fit(X_train, Y_train)
		scores = classification_report(Y_valid, clf.predict(X_valid), digits=4, output_dict=True)
		score = scores['samples avg']["f1-score"]
		logger.info(f" Scores: {score} || Param Value: {param}")
		tuned_params_CNB_scores.append(score)
		if score > max_score:
			max_score = score
			max_score_param = param

	logger.info(
		f"Best Parameter Value of {max_score_param} with a score of {max_score}")
	if max_score > 0:
		params["complement_nb"]["alpha"] = max_score_param

	logger.info("Hyperparam Tuning for QDA")
	max_score = 0
	max_score_param = 0

	for i, param in enumerate(tuned_params_QDA[0]["reg_param"]):
		clf = MultiOutputClassifier(QuadraticDiscriminantAnalysis(reg_param=param))
		clf.fit(X_train.todense(), Y_train)
		scores = classification_report(Y_valid, clf.predict(X_valid.todense()), digits=4, output_dict=True)
		score = scores['samples avg']["f1-score"]
		logger.info(f" Scores: {score} || Param Value: {param}")
		tuned_params_QDA_scores.append(score)
		if score > max_score:
			max_score = score
			max_score_param = param
	logger.info(
		f"Best Parameter Value of {max_score_param} with a score of {max_score}")
	if max_score > 0:
		params["qda"]["reg_param"] = max_score_param

	# Instatiate Models:
	log_reg_baseline = MultiOutputClassifier(LogisticRegression(**params["logreg"]))
	multinomial_nb = MultiOutputClassifier(
		MultinomialNB(**params["multinomial_nb"]))
	complement_nb = MultiOutputClassifier(ComplementNB(**params["complement_nb"]))
	svm_rbf = MultiOutputClassifier(SVC(**params["svm_rbf"]))
	# svm_polynomial = MultiOutputClassifier(SVC(**params["svm_polynomial"]))
	svm_sigmoid = MultiOutputClassifier(SVC(**params["svm_sigmoid"]))
	linear_svc = MultiOutputClassifier(LinearSVC(**params["linear_svc"]))
	lda = MultiOutputClassifier(LinearDiscriminantAnalysis())
	qda = MultiOutputClassifier(QuadraticDiscriminantAnalysis(**params["qda"]))

	baseline_classifiers = {
		"log_reg_baseline": log_reg_baseline,
		"multinomial_nb": multinomial_nb,
		"complement_nb": complement_nb,
		"svc_rbf": svm_rbf,
		"svc_sigmoid": svm_sigmoid,
		"linear_svc": linear_svc,
		"lda": lda,
		"qda": qda
	}

	# FIT
	scores_models = {}
	all_preds = np.zeros_like(Y_valid)
	for clf_name, clf in baseline_classifiers.items():
		if "svc" in clf_name:
			clf.fit(X_train, Y_train)
			predictions = clf.predict(X_valid)
			scores = classification_report(Y_valid, predictions, digits=4, output_dict=True)
		elif "qda" or "lda" in clf_name:
			clf.fit(X_train.todense(), Y_train)
			predictions = clf.predict(X_valid.todense())
			scores = classification_report(Y_valid, predictions, digits=4, output_dict=True)
		else:
			clf.fit(X_train, Y_train)
			predictions = clf.predict(X_valid)
			scores = classification_report(Y_valid, predictions, digits=4, output_dict=True)
		all_preds += predictions

		score = scores['samples avg']["f1-score"]
		logger.info(f" Scores: {score} || Param Value: {param}")
		logger.info(f" Score: {score} || Name: {clf_name}")
		scores_models[clf_name] = score

	all_preds = all_preds / len(baseline_classifiers)
	scores_models["Ensemble"] = accuracy_score(all_preds, Y_valid)

	return scores_models

config_path = "master_path//repo/ML4H_MSProject/task1/configs/baselines.jsonnet"
var = "edss_19"
baseline_labels = "master_path//repo/ML4H_MSProject/data/baseline_labels/"
score_models = run_baselines(var,config_path, baseline_labels)