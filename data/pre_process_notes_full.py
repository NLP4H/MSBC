import pandas as pd
import re

from regex_removal_functions import remove_header_footer, remove_numerical_and_dates, remove_hospitals_clinic_centre
from de_id import remove_database_info, remove_missed_names

def lower_text(df):
    notes = df.text.tolist()
    for i in range(len(notes)):
        if False == isinstance(notes[i], str):
            continue
        notes[i] = notes[i].lower()
    df['text'] = notes
    return df

def save_result(df, output_name):
    df.to_csv(output_name)

if __name__ == '__main__':
    # Load notes csv file
    import sys
    file = sys.argv[1]
    df = pd.read_csv("master_path/data/neurology_notes/raw_data/splits/"+file, error_bad_lines=False, delimiter=',')
    print(df.shape)
    df = remove_header_footer(df)
    print("Removed Header")
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
    df = remove_numerical_and_dates(df)
    print("Removed numerical")
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
    df = remove_hospitals_clinic_centre(df, "master_path/data/neurology_notes/raw_data/hostpital_list_trimmed.txt")
    print("Removed Hospitals")
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
    df = remove_database_info(df, 'master_path/data/neurology_notes/raw_data/Identification.csv')
    print("Removed Database Info")
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
    df = lower_text(df)
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
    df = remove_missed_names(df, lookup_file = 'master_path/data/neurology_notes/raw_data/missed_names.csv')
    save_result(df, "master_path/data/neurology_notes/processed_data/"+file)
