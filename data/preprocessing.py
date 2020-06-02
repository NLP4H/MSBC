import pandas as pd
import numpy as np
import random
import pdb

# sklearn imports
from sklearn.model_selection import StratifiedShuffleSplit

# our imports

def get_num_notes(df):
	if df["num_visits"] < 3:
		return 1
	else:
		return random.randint(1, df["num_visits"]-1)

def convert_int(x):
	return x.astype(int)

def index_notes(x):
	return np.arange(len(x))

def stratified_sampling(df, p, test_perc):
	"""
	Do Stratified sampling and preprocessing as discussed in the notes for EDA:
	https://docs.google.com/document/d/1Kg1AIRNsyJug6qxdF22n_Kr5Hr1IPom1Lt5Jf_lIpso/edit
	For now:
		Add a visit number column
		Do stratified sampling by gender only.
		For single visit patients assign randomly splitting by gender
		For multi-visit:
		Randomly assign to be fully contained or split across (with p = TBD)
		For fully contained just shoot off to the test based on %
		For split across - Randomly determine the number of notes to be put in Test (at least 1) 
	"""
	# Make Number of visits
	num_visits = df.groupby(by=["patient_id"], as_index=False).visit_date.agg("count") #.rename(mapper={"visit_date": "num_visits"}
	# Get per patient table
	patients = df.copy()[["patient_id", "gender"]].drop_duplicates()
	patients = patients.merge(num_visits, on = 'patient_id', how='left')
	patients.rename(columns={'visit_date':'num_visits'}, inplace=True)
	df = df.merge(num_visits, on = 'patient_id', how='left')
	df.rename(columns={'visit_date_y':'num_visits'}, inplace=True)
	df.rename(columns={'visit_date_x':'visit_date'}, inplace=True)
	patients["full_move"] = 1
	sss = StratifiedShuffleSplit(n_splits=1, test_size=p)
	# Determine which people to split
	splits = []
	for i in sss.split(patients, np.zeros(patients.shape[0]), groups=patients.gender):
		splits.append(i)
	split = splits[0][0]
	fm = splits[0][1]
	patients["full_move"].iloc[split] = 0
	# all num_visits = 1 patients are full move
	patients["full_move"].mask(patients.num_visits == 1, 1, inplace=True)
	patients["num_notes_to_move"] = patients.apply(get_num_notes, axis=1)
	# Get note indices
	df["note_index"] = -1
	df["note_index"] = df.sort_values(by = ["patient_id", "visit_date"], ascending = True).groupby("patient_id").transform(index_notes).astype(int)
	# Do Merges
	full_move = patients.loc[patients.full_move == 1]
	split = patients.loc[patients.full_move != 1]
	final = pd.merge(df, split, on="patient_id", how="right")
	splits = []
	for i in sss.split(full_move, np.zeros(full_move.shape[0]),groups=full_move.gender):
		splits.append(i)
	train = splits[0][0]
	test = splits[0][1]
	full_move["train"] = 0
	full_move["train"].iloc[train] = 1
	fm_train_df = full_move.iloc[train]
	fm_test_df = full_move.iloc[test]
	train_df = df.merge(fm_train_df, on="patient_id", how="right")
	test_df = df.merge(fm_test_df, on="patient_id", how="right")
	final["train"] = 0
	final["train"] = final.note_index < final.num_notes_to_move
	fin_train_df = pd.concat([train_df, final[final.train == True]])
	fin_test_df = pd.concat([test_df, final[final.train != True]])
	fin_train_df = fin_train_df.drop(columns=['num_visits_x', 'note_index', 'gender_y', 'num_visits_y', 'full_move', 'num_notes_to_move', 'train'])
	fin_test_df = fin_test_df.drop(columns=['num_visits_x', 'note_index', 'gender_y', 'num_visits_y', 'full_move', 'num_notes_to_move', 'train'])
	fin_train_df.rename(columns={'gender_x':'gender'}, inplace=True)
	fin_test_df.rename(columns={'gender_x':'gender'}, inplace=True)
	assert fin_train_df.shape[0] + fin_test_df.shape[0] == df.shape[0], "BAD BOI"
	return (fin_train_df, fin_test_df)

if __name__ == "__main__":
	df_test = pd.read_csv("master_path/data/neurology_notes/processed_data/train_data.csv")
	df_labels = pd.read_csv("master_path/data/neurology_notes/raw_data/r_test_labels.csv")
	''' Below is used to add labels to r_split '''
	#df_combined = pd.concat([r_test_labels, r_test_labels], axis=1, sort=False)
	train_df, test_df = stratified_sampling(df=df_test, p=0.3, test_perc=0.3 )
	train_df.to_csv('master_path/data/neurology_notes/processed_data/train_data.csv')
	test_df.to_csv('master_path/data/neurology_notes/processed_data/val_data.csv')
