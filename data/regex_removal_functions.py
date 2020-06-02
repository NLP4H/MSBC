import pandas as pd
import re
import datetime

# Remove header footer irrelevant info
def remove_header_footer(df):
    '''
    Remove note header and footer
    Input: pandas dataframe
    Output: pandas dataframe (with notes processed)
    '''
    notes = df.text.tolist()
    # Patterns
    # Header Keep, tokenize out later
    p2 = re.compile(r'(Yours sincerely|sincerely|With kind regards|kind regards|Best regards).*', re.IGNORECASE)
    p3 = re.compile(r'D:.*(A|P)?\sT\:.*')
    # Signature
    p4 = re.compile(r'([A-Z][a-z]+|[A-Z].)+\s[A-Z][a-z]+,\s(MD|M.D.)')
    p5 = re.compile(r'[A-z][a-z]+\s[A-Z][a-zA-Z]*(-[A-Z][a-zA-Z]*),\s(MD|M.D.)')
    # Remove header & footer
    for i in range(df.shape[0]):
        if False == isinstance(notes[i], str):
            continue
        if i % 10000 == 0: 
            print(str(datetime.datetime.now()) + " Processed ", i , " notes")
        # Footer
        if len(re.findall(p2, notes[i])) > 0 or len(re.findall(p3, notes[i])) > 0:
            notes[i] = re.sub(p2, r'', notes[i])
            notes[i] = re.sub(p3, r'', notes[i])
        # Signature
        if len(re.findall(p4, notes[i])) > 0 or len(re.findall(p5, notes[i])) > 0:
            notes[i] = re.sub(p4, r'', notes[i])
            notes[i] = re.sub(p5, r'', notes[i])
    # Save temporary results
    df['text'] = notes
    return df

# Remove header footer irrelevant info
def remove_numerical_and_dates(df):
    '''
    Removes any numerical, address and date information from notes
    Input: pandas dataframe
    Output: pandas dataframe (with notes processed)
    '''
    notes = df.text.tolist()
    # Patterns
    # Telephone & Fax
    p1 = re.compile(r'(phone:|fax:|tel:|#|\s)(\s?)\(?\d{3}\)?(-?|\s?)\d{3}(-?|\s?)\d{4}', re.IGNORECASE)
    # MRN
    p2 = re.compile(r'(\(?MRN\)?:?\s?\d{3}(-?)\d{4})|\d{5,6,7}|(MRN\s#\d{5,6,7})|MRN\s\"\d{8}\"')
    # D.O.B
    p3 = re.compile(r'((D.O.B.|DOB)?\s?:?\s?)(\d{1,2}-\d{1,2}-\d{4}|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}\/\d{1,2}\/\d{4}|[A-Za-z]{3}-\d{1,2}-\d{4}|\d{4}(-|\/)[A-Za-z]{3}(-|\/)\d{1,2}|[A-Z][a-z]+\s\d{1,2}, \d{4}|\d{1,2}\/\d{1,2}\/\d{2,4})', re.IGNORECASE)
    # Other dates
    p4 = re.compile(r'\d{1,2}\s(January|Jan|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December),?\s\d{4}')
    p4a = re.compile(r'(January|Jan|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December) \d{2}(rd|st|nd|th)', re.IGNORECASE)
    p4b = re.compile(r'(January|Jan|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December) \d{4}', re.IGNORECASE)
    # Time
    p5 = re.compile(r'(\d{2}:\d{2})|(\d{2}:\d{2}:\d{2})')
    # Address
    p6 = re.compile(r'\d{1,3}.?\d{0,3}\s[a-zA-Z]{2,30}\s(Street|St.|Avenue|Ave.)') # Street No. and Name
    p7 = re.compile(r'(Toronto,)?\sON\s[A-Z]\d[A-Z]\s?\d[A-Z]\d', re.IGNORECASE) # "City & Postal code"
    # cc: and make sure it's at the end of the sentence
    p8 = re.compile(r'cc:.{0,1000}$')
    for i in range(len(notes)):
        if False == isinstance(notes[i], str):
            continue
        if i % 10000 == 0: 
            print(str(datetime.datetime.now()) + " Processed ", i , " notes")
        # Phone/Fax
        if len(re.findall(p1, notes[i])) > 0:
            notes[i] = re.sub(p1, ' 1718 ', notes[i])
        # MRN#
        if len(re.findall(p2, notes[i])) > 0 or len(re.findall(r'Patient Identifier', notes[i])) > 0:
            notes[i] = re.sub(p2, ' 999 ', notes[i])
            notes[i] = re.sub(r'Patient Identifier', ' 999 ', notes[i])
        # DOB
        if len(re.findall(p3, notes[i])) > 0 or len(re.findall(r'Date of Birth \(DOB\) \((MON dd, yyyy|yyyy-mm-dd)\)|Date of Birth \(DOB\)|DATE OF BIRTH', notes[i])) > 0:
            notes[i] = re.sub(p3, ' 2010s ', notes[i])
            notes[i] = re.sub(r'Date of Birth \(DOB\) \((MON dd, yyyy|yyyy-mm-dd)\)|Date of Birth \(DOB\)|DATE OF BIRTH', ' 2010s ', notes[i])
        # Other dates
        if len(re.findall(p4, notes[i])) > 0:
            notes[i] = re.sub(p4, ' 2010s ', notes[i])
        if len(re.findall(p4a, notes[i])) > 0:
            notes[i] = re.sub(p4a, ' 2010s ', notes[i])
        if len(re.findall(p4b, notes[i])) > 0:
            notes[i] = re.sub(p4b, ' 2010s ', notes[i])
        # Time
        if len(re.findall(p5, notes[i])) > 0:
            notes[i] = re.sub(p5, ' 1610 ', notes[i])
        # Address info
        if len(re.findall(p6, notes[i])) > 0 or len(re.findall(p7, notes[i])) > 0:
            notes[i] = re.sub(p6, ' silesia ', notes[i])
            notes[i] = re.sub(p7, ' silesia ', notes[i])
        # Check whether 'cc:' is in the footer & remove everything after 'cc:'
        if len(re.findall(p8, notes[i])) > 0:
            notes[i] = re.sub(r'cc:.*', r'', notes[i])
        if len(re.findall(r'CC:.{0,1000}$', notes[i])) > 0:
            notes[i] = re.sub(r'CC:.*', r'', notes[i])
    df['text'] = notes
    return df

def remove_hospitals_clinic_centre(df, file_name):
    notes = df.text.tolist()
    known_hospitals = {}
    with open(file_name) as in_hospitals:
        for line in in_hospitals:
            known_hospitals[line.strip().lower()] = 0
    for i in range(len(notes)):
        if False == isinstance(notes[i], str):
            continue
        if i % 10000 == 0: 
            print(str(datetime.datetime.now()) + " Processed ", i , " notes")
        for name in known_hospitals:
            n_hospital_a = name + "hospital"
            n_hospital_b = name + "Hospital"
            if len(re.findall(n_hospital_a, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_hospital_a, ' troy hospital ', notes[i], flags = re.IGNORECASE)
            elif len(re.findall(n_hospital_b, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_hospital_b, ' troy hospital ', notes[i], flags = re.IGNORECASE)
            n_clinic_a = name + "clinic"
            n_clinic_b = name + "Clinic"
            if len(re.findall(n_clinic_a, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_clinic_a, ' troy clinic ', notes[i], flags = re.IGNORECASE)
            elif len(re.findall(n_clinic_b, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_clinic_b, ' troy clinic ', notes[i], flags = re.IGNORECASE)
            n_centre_a = name + "centre"
            n_centre_b = name + "Centre"
            if len(re.findall(n_centre_a, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_centre_a, ' troy centre ', notes[i], flags = re.IGNORECASE)
            elif len(re.findall(n_centre_b, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(n_centre_b, ' troy centre ', notes[i], flags = re.IGNORECASE)
            if len(re.findall(name, notes[i], flags = re.IGNORECASE)) > 0:
                notes[i] = re.sub(name, ' troy ', notes[i], flags = re.IGNORECASE)
    df['text'] = notes
    return df


if __name__ == '__main__':
    # Load notes csv file
    df = pd.read_csv("neurology_notes.csv", error_bad_lines=False, delimiter=',')
    print(df.shape)
    df = remove_header_footer(df)
    df = remove_numerical_and_dates(df)
    df = remove_hospitals_clinic_centre(df, "master_path/data/neurology_notes/raw_data/hostpital_list_trimmed.txt")

