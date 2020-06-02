import pandas as pd 
df = pd.read_csv('master_path/data/neurology_notes/processed_data/neurology_notes_tokenized.csv')
print(df.keys())
# examples
ex_df = pd.read_csv('master_path/data/neurology_notes/processed_data/Final Splits/EDA_data_w_tokens.csv')
print(ex_df.keys())
new_df = pd.DataFrame(columns = ['patient_id', 'visit_date', 'status', 'first_visit',
       'administrative', 'routine', 'suspected_relapse', 'examined_by',
       'edss_19', 'edss_10', 'edss_4', 'edss_3', 'gender',
       'score_brain_stem_subscore', 'score_cerebellar_subscore',
       'score_mental_subscore', 'score_visual_subscore',
       'score_pyramidal_subscore', 'score_sensory_subscore',
       'score_bowel_bladder_subscore', 'score_ambulation_subscore',
       'score_brain_stem_subscore_binary', 'score_cerebellar_subscore_binary',
       'score_mental_subscore_binary', 'score_visual_subscore_binary',
       'score_pyramidal_subscore_binary', 'score_sensory_subscore_binary',
       'score_bowel_bladder_subscore_binary',
       'score_ambulation_subscore_binary', 'employment_status',
       'employment_status_binary', 'living_arrangement',
       'living_arrangement_binary', 'regular_cigarette_use',
       'regular_cigarette_use_binary', 'regular_alcohol_use',
       'regular_alcohol_use_binary', 'text', 'ms_type', 'recent_relapse',
       'future_relapse', 'Birth Date', 'age', 'note_index', 'full_move',
       'num_notes_to_move', 'tokenized_text', 'split'])
new_df[['patient_id', 'visit_date', 'text', 'tokenized_text']] = df[['patient_id', 'ObservationDateTime', 'text', 'tokenized_text']]
empty_cols = [col for col in new_df.columns if new_df[col].isnull().all()]
new_df[empty_cols] = -1 
cond = new_df['text'].isin(ex_df['text'])
new_df.drop(new_df[cond].index, inplace = True)
print(len(new_df))
new_df.to_csv('master_path/data/neurology_notes/processed_data/Final Splits/unlabeled_data.csv')
