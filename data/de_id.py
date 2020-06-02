import pandas as pd
import numpy as np
import math 
import string
import re
import datetime

# Remove any PHI info 
def remove_database_info(df, lookup_file):
    '''
    Looks up words from CSV and replaces them with a token 
    '''
    #load note
    notes = df.text.tolist()
    genders = []
    if 'gender' in df.columns:
        genders = df.gender.tolist()
    else:
        # Lookup gender based on the lookup_file
        lookup_df = pd.read_csv(lookup_file)
        for i in range(df.shape[0]):
            if df.MRN[i] in lookup_df['Patient Code']:
                genders.append(lookup_df[lookup_df['Patient Code'] == df.MRN[i]]['Gender'].astype(str).item())
            else:
                # Code not found, asssume female
                genders.append('F')

    # PHI Coloumns Names 
    phi_types = ['Patient Code', 'Last Name', 'First Name', 'Birth Date', 'Gender', \
        'Birth Country', 'Zip Code', 'State', 'Doctor in Charge', 'Referred By Who']

    lookup_df = pd.read_csv(lookup_file, usecols = phi_types)
    lookup_dict = lookup_df.to_dict()

    # load and lower last names
    l_name_list = lookup_dict['Last Name'].values()
    l_name_list = [x.lower() for x in l_name_list]

    # define list of names we do not to remove 
    non_name_list = ['to', 'of', 'or', 'at', 'is', 'scan', 'last', 'brain', 'MRI', 
        'hand', 'oh', 'long', 'back', 'March', 'walker', 'lower', 'see', 'able', 'power', 'White', 'long', 'light', 
        'field', 'keen', 'then', 'fine', 'little', 'fear', 'hall', 'day']

    # remove the names that are in the non name list
    for non_name in non_name_list:
        if non_name in l_name_list: 
            l_name_list.remove(non_name)

    l_name_compile = {}
    for name in l_name_list:
        l_name_compile[name] = re.compile(rf"(?<!\w){name}(?!\w)", re.IGNORECASE)

    # load first name list, patient code name, refence name list, and list of birth coutnries 
    f_name_list = lookup_dict['First Name'].values()
    f_name_compile = {}
    for name in f_name_list:
        f_name_compile[name] = re.compile(rf"(?<!\w){name}(?!\w)", re.IGNORECASE)
    ref_list = lookup_dict['Referred By Who'].values()
    ref_compile = {}
    for ref in ref_list:
        # check to see it is not a nan in list
        if False == isinstance(ref, str):
            continue
        # names are stored as "John Smith" thus split and remove non-names 
        names = ref.split(' ')
        for non_name in non_name_list:
            if non_name in names: 
                names.remove(non_name) 
                names.append("place_holder_name") 
        # remove the fist name 
        ref_compile[names[0]] = re.compile(rf"(?<!\w){names[0]}(?!\w)")
        ref_compile[names[1]] = re.compile(rf"(?<!\w){names[1]}(?!\w)")
    b_country_compile = {}
    b_country_list = lookup_dict['Birth Country'].values()
    for b_land in b_country_list:
        # check for nans 
        if False == isinstance(b_land, str):
            continue
        b_country_compile[b_land] = re.compile(rf"\<(?=\w){b_land}\>(?!\w)")
    
    r9 = re.compile(r'(?:Mr\. | Ms\. | Dr\.|Yours Sincerly\,|Sincerely\, | truely\,) [a-zA-Z]+', re.IGNORECASE)
    r10 = re.compile(r'(?:Mr\. | Ms\. | Dr\.|Yours Sincerly\,|Sincerely\,| truely\,)(| |  |   )[a-zA-Z]+(| |  |   )\b[A-Z].*?\b')
    #first 10 notes for test 
    for i in range(len(df)): #len(notes)):
        if False == isinstance(notes[i], str):
            continue
        if i % 100 == 0: 
            print(str(datetime.datetime.now()) + " Processed ", i , " notes")
        note = notes[i]
        for name in l_name_list:
            note = re.sub(l_name_compile[name], "Salamanca", note) 
        
        if genders[i] == 'F':
            for name in f_name_list:
                note = re.sub(f_name_compile[name], "Lucie", note)
        else:
            for name in f_name_list:
                note = re.sub(f_name_compile[name], "Ezekiel", note) 
        
        for ref in ref_list:
            # check to see it is not a nan in list
            if False == isinstance(ref, str):
                continue
            # names are stored as "John Smith" thus split and remove non-names 
            names = ref.split(' ')
            for non_name in non_name_list:
                if non_name in names: 
                    names.remove(non_name) 
                    names.append("place_holder_name") 
            note = re.sub(ref_compile[names[0]], "Ezekiel", note) 
            note = re.sub(ref_compile[names[1]], "Salamanca", note)

        for b_land in b_country_list:
            # check for nans 
            if False == isinstance(b_land, str):
                continue
            if len(re.findall(b_country_compile[b_land], note)) > 0:
                note = re.sub(b_country_compile[b_land], "Madagascar", note)
                break
        
        ''' This code is case sensative'''
        # Get rid of remaning names  
        note = re.sub(r9, " Dr. Salamanca ", note)
        # Check for last names based on captialization 
        note = re.sub(r10, " Dr. Ezekiel Salamanca ", note) 

        # Update the note 
        notes[i] = note
    # Update dataframe
    df['text'] = notes
    return df

def remove_missed_names(df, lookup_file):
    notes = df.text.tolist()
    lookup_df = pd.read_csv(lookup_file)
    lookup_dict = lookup_df.to_dict()
    missed_name_list = lookup_dict['names'].values()
    print(missed_name_list)

    for i in range(len(df)):
        note = notes[i]
        if False == isinstance(note, str):
            continue
        for name in missed_name_list:
            r1 = re.compile(rf"(?<!\w){name}(?!\w)", re.IGNORECASE)
            note = re.sub(r1, "Salamanca", note) 
            
        notes[i] = note
    # Update dataframe
    df['text'] = notes
    return(df)

if __name__ == '__main__':
        # Test
        print("Processing raw data")
        # Load psv file
        df = pd.read_csv("master_path/data/neurology_notes/raw_data/r_test_notes.csv", error_bad_lines=False, delimiter=',')
        print(df.shape)
        df = remove_header_footer(df)
        df = remove_numerical_and_dates(df)
        df = remove_database_info(df)