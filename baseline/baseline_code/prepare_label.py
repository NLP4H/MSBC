import tensorflow as tf
print(tf.__version__)
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.utils import class_weight
import pandas as pd
import os
import pickle

def label_transform(variable_name, labels):
    
    """
    Turn labels into categorical data
    Input: 
    - variable_name -> the variable of interest(i.e. "edss_19")
    - labels -> observations (i.e. all training samples edss_19 score)
    Output: Matrix (Number of data * Number of categories)
    
    """
    # Set a default value for number of classes
    num_classes = len(list(set(labels)))
    converted_labels = []

    # Reason for visit (categories)
    if variable_name == "status":
        """
        Class info:
        0: First Visit
        1: Administrative
        2: Routine
        3: Suspected Relapse
        4: Being assigned multiple labels

        """
        num_classes = 5
        for i in range(len(labels)):
            converted_labels.append(labels[i] - 1) 

    # Examine clinician
    if variable_name == "examined_by":
        
        """
        Class info: 
        0: Alexandra Roll
        1: Chantal Roy-Hewitson
        2: Dale Robinson
        3: Dalia Rotstein
        4: Daniel Selchen
        5: David Morgenthau
        6: Jiwon Oh
        7: Marika Hohol
        8: Paul Marchetti
        9: Reza Vosoughi
        10: Xavier Montalban
        11: Unknown
        
        """
        num_classes = 12
        for i in range(len(labels)):
            if labels[i] != -1:
                converted_labels.append(labels[i] - 1)
            else:
                converted_labels.append(11)

    # EDSS 19 classs
    if variable_name == "edss_19":
        num_classes = 20
        label_dict = {0.0:0,
                      1.0:1,
                      1.5:2,
                      2.0:3,
                      2.5:4,
                      3.0:5,
                      3.5:6,
                      4.0:7,
                      4.5:8,
                      5.0:9,
                      5.5:10,
                      6.0:11,
                      6.5:12,
                      7.0:13,
                      7.5:14,
                      8.0:15,
                      8.5:16,
                      9.0:17,
                      9.5:18,
                      -1:19}
        for i in range(len(labels)):
            converted_labels.append(label_dict[labels[i]])       
    
    # EDSS 10 classes
    if variable_name == "edss_10":
        num_classes = 11
        converted_labels = np.array(labels)
        # Replace -1 to the 10th class
        converted_labels[converted_labels == -1] = 10
    
    # EDSS 4 classes
    if variable_name == "edss_4":

        """
        Class info: 
        0: EDSS 0
        1: EDSS 1.0 to 4.0
        2: EDSS 4.5 to 5.5
        3: EDSS 6.0 to 9.5
        4: Unknown
        """
        num_classes = 5
        for i in range(len(labels)):
            if labels[i] != -1:
                converted_labels.append(labels[i] - 1)
            else:
                converted_labels.append(4)

    # EDSS 3 classes  
    if variable_name == "edss_3":

        """
        Class info: 
        0: EDSS 0
        1: EDSS 1.0 to 5.5
        2: EDSS 6.0 to 9.5
        3: Unknown
        """
        num_classes = 4
        for i in range(len(labels)):
            if labels[i] != -1:
                converted_labels.append(labels[i] - 1)
            else:
                converted_labels.append(3)

    # Living Arrangement
    if variable_name == "living_arrangement":

        """
        Class info:
        0: Relatives
        1: Alone
        2: Caregivers
        3: Unknown
        """
        num_classes = 4
        for i in range(len(labels)):
            if labels[i] != -1:
                converted_labels.append(labels[i] - 1)
            else:
                converted_labels.append(3)

    # Employment Status
    if variable_name == "employment_status":
        
        """
        Class info:
        0: Full time
        1: Disability
        2: Unemployed
        3: Part time
        4: Retired
        5: Student
        6: Leave
        7: Maternity
        8: Unknown
        """
        num_classes = 9
        for i in range(len(labels)):
            if labels[i] != -1:
                converted_labels.append(labels[i] - 1)
            else:
                converted_labels.append(8)

    # Subscore: Brainstem
    if variable_name == "score_brain_stem_subscore":
        num_classes = 7
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 6
    
    # Subscore: Cerebellar
    if variable_name == "score_cerebellar_subscore":
        num_classes = 7
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 6
    
    # Subscore: Mental
    if variable_name == "score_mental_subscore":
        num_classes = 7
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 6

    # Subscore: Visual
    if variable_name == "score_visual_subscore":
        num_classes = 8
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 7
    
    # Subscore: Pyramidal
    if variable_name == "score_pyramidal_subscore":
        num_classes = 8
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 7
    
    # Subscore: Sensory
    if variable_name == "score_sensory_subscore":
        num_classes = 8
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 7
    
    # Subscore: Bowel Bladder
    if variable_name == "score_bowel_bladder_subscore":
        num_classes = 8
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 7

    # Subscore: Ambulation
    if variable_name == "score_ambulation_subscore":
        num_classes = 16
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 15

    # Interim worsening
    if variable_name in ["regular_cigarette_use", "regular_alcohol_use"]:
        num_classes = 3
        converted_labels = np.array(labels)
        converted_labels[converted_labels == -1] = 2

    # MRI Y/N/Unknown variables
    if variable_name in ["mri_worsening", "enhancing_lesion"]:
        num_classes = 3
        for i in range(len(labels)):
            if labels[i] == "No":
                converted_labels.append(0)
            if labels[i] == "Yes":
                converted_labels.append(1)
            if labels[i] == -1:
                converted_labels.append(2)
    
    # Number of new T2 lesions, change to binary
    if variable_name == "num_new_t2_lesions":
        """
        Class info:
        0: 0 lesions
        1: Above 0 lesions
        2: Unknown
        """
        num_classes = 3
        for i in range(len(labels)):
            if labels[i] == 0:
                converted_labels.append(0)
            if labels[i] > 0:
                converted_labels.append(1)
            if labels[i] == -1:
                converted_labels.append(2)


    # Categorize converted labels
    converted_labels = np.array(converted_labels)
    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(converted_labels), converted_labels)
    categorical_labels = to_categorical(converted_labels, num_classes=num_classes)

    return class_weights, categorical_labels

# TODO
def label_reverse(variable_name, class_name):
    
    """
    Turn model predicted variables into it's original format
    Input: Variable name (str); class_name (int)
    Return: Readable predicted variable value (str)
    """

    prediction = ""
    
    # Reason for visit (categories)
    if variable_name == "status":
        label_dict = {
            0: "First Visit",
            1: "Administrative",
            2: "Routine",
            3: "Suspected Relapse",
            4: "Need to reconfirm: Being assigned multiple labels"}
    
    # Examine clinician
    if variable_name == "examined_by":
        label_dict = {
            0: "Alexandra Roll",
            1: "Chantal Roy-Hewitson",
            2: "Dale Robinson",
            3: "Dalia Rotstein",
            4: "Daniel Selchen",
            5: "David Morgenthau",
            6: "Jiwon Oh",
            7: "Marika Hohol",
            8: "Paul Marchetti",
            9: "Reza Vosoughi",
            10: "Xavier Montalban",
            11: "Unknown"}
        prediction = label_dict[class_name]
    
    # EDSS 19 classes
    if variable_name == "edss_19":
        label_dict = {
            0: "0.0",
            1: "1.0",
            2: "1.5",
            3: "2.0",
            4: "2.5",
            5: "3.0",
            6: "3.5",
            7: "4.0",
            8: "4.5",
            9: "5.0",
            10: "5.5",
            11: "6.0",
            12: "6.5",
            13: "7.0",
            14: "7.5",
            15: "8.0",
            16: "8.5",
            17: "9.0",
            18: "9.5",
            19: "Unknown"}
        prediction = label_dict[class_name]
    
    # EDSS 19 classes
    if variable_name == "edss_10":
        if class_name == 10:
            prediction = "Unknown"
        else:
            prediction = str(class_name)
    # TODO Subscores



    # TODO: Patient characteristics ?
    # e.g. dominant hand
    return prediction


def load_label(train_dir, val_dir, test_dir, save_dir, variable_names, note_column = "text"):
    
    """
    Input: 
    variable_names: A list of clinial variables needed

    """
    
    # Read train file
    df_train = pd.read_csv(train_dir)
    # Remove places where there's no note
    df_train = df_train.dropna(subset = [note_column])
    # Fill NA values to -1
    df_train = df_train.fillna(-1)
    df_train = df_train[df_train.edss_19 != -1]
    # Reset Index
    df_train = df_train.reset_index()
    
    # Read validation file
    df_val = pd.read_csv(val_dir)
    # Remove places where there's no note
    df_val = df_val.dropna(subset = [note_column])
    # Fill NA values to -1
    df_val = df_val.fillna(-1)
    df_val = df_val[df_val.edss_19 != -1]
    # Reset Index
    df_val = df_val.reset_index()

    # Read test file
    df_test = pd.read_csv(test_dir)
    # Remove places where there's no note
    df_test = df_test.dropna(subset = [note_column])
    # Fill NA values to -1
    df_test = df_test.fillna(-1)
    df_test = df_test[df_test.edss_19 != -1]
    # Reset Index
    df_test = df_test.reset_index()
    
    
    for var in variable_names:
        print("Creating label for: ", var)
        class_weights, y_train = label_transform(var, list(df_train[var]))

        # print class weights
        print("Class weights for: ", var)
        print(class_weights)

        # Save class weights
        with open(var + "_class_weights.pickle", 'wb') as handle:
            pickle.dump(class_weights, handle, protocol = pickle.HIGHEST_PROTOCOL)

        _, y_val = label_transform(var, list(df_val[var]))
        _, y_test = label_transform(var, list(df_test[var]))


        # Save labels
        print("Saving files")
        np.savez(var + ".npz", y_train, y_val, y_test)

    return 
    
def prep_label(save_dir_intermediate_data, train_dir, val_dir, test_dir, mri_load, neurology_load):
    print("running prepare_label.py")
    save_dir = save_dir_intermediate_data
    # Save data
    os.chdir(save_dir)
    # ---------------------------------------- NEUROLOGY ---------------------------------------------------------------------------
    if(neurology_load):
        print("Preparing visit level labels")
        # Save labels as intermediate files
        visit_level_variables = ["edss_19",
                                "score_brain_stem_subscore", 
                                "score_cerebellar_subscore", 
                                "score_pyramidal_subscore", 
                                "score_sensory_subscore",
                                "score_bowel_bladder_subscore", 
                                "score_ambulation_subscore", 
                                "score_mental_subscore", 
                                "score_visual_subscore", 
                                "examined_by", 
                                "living_arrangement", 
                                "status", 
                                "employment_status", 
                                "regular_cigarette_use", 
                                "regular_alcohol_use"]

        train_dir = train_dir
        val_dir = val_dir
        test_dir = test_dir
        
        load_label(train_dir, val_dir, test_dir, save_dir, visit_level_variables, note_column = "text")


    # ---------------------------------------- RADIOLOGY ---------------------------------------------------------------------------
    if(mri_load):
        print("Preparing MRI labels")
        mri_variables = ["mri_worsening", 
                        "enhancing_lesion", 
                        "num_new_t2_lesions"]
        train_dir = train_dir
        val_dir = val_dir
        load_label(train_dir, val_dir, save_dir, mri_variables, note_column = "result")