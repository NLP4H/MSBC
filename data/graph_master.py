import pandas as pd
import numpy as np
import random
import pdb
from datetime import timedelta
#import transformers 
#from transformers import BertModel, BertTokenizer
import torch
import numpy as np


# sklearn imports
from sklearn.model_selection import StratifiedShuffleSplit
random.seed(42069)
# our imports

def get_num_notes(df):
	if df["num_visits"] < 3:
		return 1
	else:
		return random.randint(2, df["num_visits"]-1)

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
	print(patients.shape[0])
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

	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_perc)
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
	split_train_df = final[final['train'] == True]
	split_test_df = final[final.train != True]

	# formatting
	fin_train_df = fin_train_df.drop(columns=['num_visits_x', 'gender_y', 'num_visits_y', 'train'])
	fin_test_df = fin_test_df.drop(columns=['num_visits_x', 'gender_y', 'num_visits_y', 'train'])
	fin_train_df.rename(columns={'gender_x':'gender'}, inplace=True)
	fin_test_df.rename(columns={'gender_x':'gender'}, inplace=True)


	assert fin_train_df.shape[0] + fin_test_df.shape[0] == df.shape[0], "BAD BOI"
	return (fin_train_df, fin_test_df)

def add_more_labels(main_df):
	main_df['visit_date'] =  pd.to_datetime(main_df['visit_date'], format='%Y/%m/%d')

	id_df = pd.read_csv("master_path/data/neurology_notes/raw_data/Identification.csv")
	ms_courses = id_df[['Patient ID', 'MSCourse1', 'Date MSCourse 1', 'MSCourse2', 'Date MSCourse 2', 'MSCourse3', 'Date MSCourse 3']]
	ms_courses.rename(columns={'Patient ID':'patient_id'}, inplace=True)
	total_df = main_df.merge(ms_courses, on="patient_id", how="left")

	total_df['date_c_1'] =  pd.to_datetime(total_df['Date MSCourse 1'], format='%d.%m.%Y')
	total_df['date_c_2'] =  pd.to_datetime(total_df['Date MSCourse 2'], format='%d.%m.%Y')
	total_df['date_c_3'] =  pd.to_datetime(total_df['Date MSCourse 3'], format='%d.%m.%Y')

	# now every note should have that paitents 
	# corresponding patients MS history
	total_df['ms_type'] = 0 
	for index, row in total_df.iterrows():
		if (row.visit_date >= row.date_c_3): 
			total_df.ms_type[index] = row['MSCourse3']
		if (((row.visit_date < row.date_c_3) and (row.visit_date >= row.date_c_2)) or (pd.isnull(row['MSCourse3']))):
			total_df.ms_type[index] = row['MSCourse2']
		if (((row.visit_date < row.date_c_2) and (row.visit_date >= row.date_c_1)) or (pd.isnull(row['MSCourse2']))):
			total_df.ms_type[index] = row['MSCourse1']
		if (pd.isnull(row.date_c_1) ): 
			total_df.ms_type[index] = "NA"

	total_df = total_df.drop(columns=['MSCourse1', 'Date MSCourse 1',  'MSCourse2',  'Date MSCourse 2',  'MSCourse3',  'Date MSCourse 3',   'date_c_1',   'date_c_2',   'date_c_3'])
	
	replase = pd.read_csv("master_path/data/neurology_notes/raw_data/relapse.csv")
	replase = replase[['Patient ID', 'Relapse Date']]
	replase.rename(columns={'Patient ID':'patient_id'}, inplace=True)
	#re_dates = replase.groupby('patient_id')
	replase['re_date'] =  pd.to_datetime(replase['Relapse Date'], format='%d.%m.%Y')
	re_dates = replase.groupby('patient_id')['re_date'].apply(list)
	#print(re_dates)
	print(re_dates.head(15))
	print(re_dates.tail(15))
	print(total_df.head(15))
	print(total_df.tail(15))
	fin_df = total_df.merge(re_dates, on="patient_id", how="right")
	#print(fin_df)
	fin_df['recent_relapse'] = 0 
	fin_df['future_relapse'] = 0 
	for index, row in fin_df.iterrows():
		list_dates = row.re_date
		for relapse_date in list_dates:
			if ((row.visit_date - timedelta(days=365)) < relapse_date) and (relapse_date < row.visit_date):
				fin_df.recent_relapse[index] = 1
			if ((row.visit_date + timedelta(days=365)) > relapse_date) and (relapse_date > row.visit_date):
				fin_df.future_relapse[index] = 1

	#print(fin_df['recent_relapse'].head(10))
	#print(fin_df['future_relapse'].head(10))
	#len(fin_df[fin_df['recent_relapse'] == 1])
	fin_df = fin_df.drop(columns=['re_date'])
	#print(fin_df)

	return fin_df

def add_age(df):
	total_df = df
	# load list of birthdays for each patient 
	id_df = pd.read_csv("master_path/data/neurology_notes/raw_data/Identification.csv")
	date_of_birth = id_df[['Birth Date', 'Patient ID']]
	date_of_birth.rename(columns={'Patient ID':'patient_id'}, inplace=True)

	# calculate age for everyone 

	total_df = total_df.merge(date_of_birth, on="patient_id", how="right")
	
	total_df['visit_date'] =  pd.to_datetime(total_df['visit_date'], format='%Y/%m/%d')
	total_df['birth_date'] =  pd.to_datetime(total_df['Birth Date'], format='%d.%m.%Y')
	total_df['birth_date'] = total_df['birth_date'].where(total_df['birth_date'] < total_df['visit_date'], total_df['birth_date'] -  np.timedelta64(100, 'Y'))
	total_df['age'] = (total_df['visit_date']- total_df['birth_date']).astype('timedelta64[Y]') 
	total_df = total_df.drop(columns=['birth_date'])
	return total_df


def tokenize_that_df(df):
	# If the tokenizer uses a single vocabulary file
	tokenizer = BertTokenizer.from_pretrained('master_path/models/base_blue_bert_pt/vocab.txt')
	print(df['text'].head())
	df['text'] = df.text.fillna('')
	#replace new lines
	df = df.replace({r'\s+$': '', r'^\s+': ''}, regex = True).replace(r'\n', ' ', regex = True)
	# generates tokens 
	token_list = df.apply(lambda x: tokenizer.encode(x['text'], add_special_tokens=True), axis = 1) 
	#join on tabs
	token_list = token_list.astype(str)
	print(token_list)
	df['tokenized_text'] = token_list

	return(df)

def create_training_test():
	df_test = pd.read_csv("master_path/data/neurology_notes/processed_data/r_test_notes_processed_removed_names.csv")
	df_labels = pd.read_csv("master_path/data/neurology_notes/raw_data/r_test_labels.csv")
	df_combined = pd.concat([df_labels, df_test.text], axis=1, sort=False)
	# add additional columns for MS types and replase 
	df_combined = add_more_labels(df_combined)
	df_combined = add_age(df_combined)
	print(df_combined.keys())
	train_df, test_df = stratified_sampling(df=df_combined, p=0.5, test_perc=0.2)
	return(train_df, test_df)

def create_training_val(full_train_df):
	full_train_df = full_train_df.drop(columns=['full_move', 'num_notes_to_move'])
	print(full_train_df.keys())
	train_df, val_df = stratified_sampling(df=full_train_df, p=0.5, test_perc=0.2)
	return(train_df, val_df)

def save_for_eda(test_df, train_df, val_df):

	# join all dataframes, and keep track of split 
	test_df['split'] = 'test'
	train_df['split'] = 'train'
	val_df['split'] = 'val'
	total_df = pd.concat([train_df, val_df, test_df])
	print(total_df.shape)

	total_df.to_csv("master_path/data/neurology_notes/processed_data/Final Splits/EDA_data_w_tokens.csv")


	print('I ran!')
	# full_test, full_train, full_val, split_test_train, split_val_test, split_train_val, split_test_train_val


if __name__ == "__main__":
	full_train_df, test_df = create_training_test()
	train_df, val_df = create_training_val(full_train_df)

	print(train_df.shape)
	print(val_df.shape)
	print(test_df.shape)

	#train_df = tokenize_that_df(train_df)
	#val_df = tokenize_that_df(val_df)
	#test_df = tokenize_that_df(test_df)

	#save_for_eda(test_df, train_df, val_df)

	#train_df.to_csv("master_path/data/neurology_notes/processed_data/Final Splits/train_data.csv")
	#val_df.to_csv("master_path/data/neurology_notes/processed_data/Final Splits/val_data.csv")
	#test_df.to_csv("master_path/data/neurology_notes/processed_data/Final Splits/test_data.csv")


