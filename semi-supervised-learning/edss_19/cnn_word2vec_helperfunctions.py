# Helper functions to preprocess the note, tokenize the note, text_to_sequence
import pandas as pd
import numpy as np
import os
from os import path

# will need to use GPU if training model file - uncomment line below
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import re
import tensorflow as tf
import string
#keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Input, Flatten, Concatenate, Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
#gensim
from gensim.models import Word2Vec, keyedvectors
import pickle
from collections import defaultdict
#nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
#sklearn
from sklearn.utils import class_weight


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
    note = " ".join(filtered)
    return note

# create tokenizer from training .csv and saves word_index and tokenizer for future use
def create_tokenizer(train_dir, save_dir):
    """
    Creates tokenizer and word_index file for CNN Word2Vec model
    Input:
        train_data_path is the path to the training .csv which contains all training notes
        save_dir is the path to save the word_index and fitted tokenizer
    Output:
        saved file: tokenizer & word index
    """ 

    # Read train file
    df_train = pd.read_csv(train_dir)
    # Remove places where there's no note
    df_train = df_train.dropna(subset = ['text'])
    # Fill NA values in labels to -1
    df_train = df_train.fillna(-1)
    # Reset Index
    df_train = df_train.reset_index()
    # We tokenize just the notes
    notes = list(df_train['text'])

    # Text level pre-processing
    texts = []
    for i in range(len(notes)):
        texts.append(tokenize_neurology(notes[i]))

    #create tokenizer 
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    # save word_indexes for embedding purposes
    word_index = tokenizer.word_index
    # print('Found %s unique tokens.' % len(word_index))

    # Save tokenizer and word index
    with open(path.join(save_dir,'tokenizer_neurology.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
    with open(path.join(save_dir, 'word_index_neurology.pickle'), 'wb') as handle:
        pickle.dump(word_index, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    return 

# runs tokenize_neurology -> tokenizes -> pads sequence to target input length for cnn model
def notes_to_sequences(note, save_dir):
    """
    Turn text into sequence matrix
    Input: 
        note - neurology note
        save_dir - directory of tokenizer (see create_tokenizer if not created already)
    Output: 
        sequence matrix
    """ 
    # PARAMS
    max_seq_length = 1000

    # Text level pre-processing (tokenizes and removes punctuation and stop words)
    text = tokenize_neurology(note)

    # Load tokenizer
    try:
        with open(path.join(save_dir, "tokenizer_neurology.pickle"), "rb") as handle:
            tokenizer = pickle.load(handle)
        # Pad sequences to text
        sequences = tokenizer.texts_to_sequences([text])
        data = pad_sequences(sequences, maxlen=max_seq_length)
    except:
        print("Error occured: did you generate tokenizer_neurology.pickle from create_tokenizer?")
    
    return data

# create word2vec embedding model
def train_word_embeddings(notes_path, save_dir):
    """
    Creates word2vec embedding model
    Input: 
        neurology note path 
        save directory path for embedding model
    Output: 
    Saves file: embedding dictionary for vocab
    """
    # Load all available notes
    print("reading input file")
    df = pd.read_csv(notes_path)

    # Remove irrelevant notes
    # Notes that should be excluded: Operative Summary, D/C Summary, eAdmit, Input Consult, ER Dpt Consult
    #                                Transfer Summary, Lithotripsy Rep, D/C Summ-Letter, Delivery Summary
    include_list = ['Amb. Consult',
                    'AmbProgressSumm',
                    'MartinFamArthrit',
                    'Mental Health',
                    'Office Consult',
                    'OfficeProgressSu',
                    'Other']

    df = df.loc[df['FindingAbbreviation'].isin(include_list)]
    
    print("text preprocessing")
    # Text Cleaning
    notes = df["text"].tolist()
    texts = []
    for i in range(df.shape[0]):
        note = tokenize_neurology(notes[i])
        note = note.split()
        texts.append(note)

    print("training word2vec model")
    # Train Word2Vec model
    model = Word2Vec(texts, size=200, window=10, min_count=2, workers=10, iter=10)

    # Save word vectors to dictionary
    vocab = []
    for word in model.wv.vocab.keys():
        vocab.append(word)
    word_vectors = model.wv
    embeddings_index = {}
    for word in vocab:
        embeddings_index[word] = word_vectors[word]

    print("Saving embeddings")
    # Save into dictionary
    with open(path.join(save_dir, 'ms_word2vec.pickle'), 'wb') as handle:
        pickle.dump(embeddings_index, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    return

# define cnn model using Word2Vec embeddings
def model_cnn(y_train, var, embedding_dim, max_sequence_length, word_index, embeddings_index, save_dir):
    """
    creates a CNN model for a particular variable (running 'create_cnn_model' calls this function)
    input:
        save_dir -> directory to save model file
        var -> edss_19 or a subscore
        embedding_dim -> the dimension of word2vec embeddings
        max_sequence_length -> max length of texts_to_sequence input for CNN
        word_index -> contains the words vocabular 
        embeddings_index -> contains the respective embedding sequence for each word within word_index
    output:
    save file: var.h5 CNN model
    """    
    # Embedding layer
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    missing = []
    count = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            count += 1
            missing.append(word)

    # define embedding layer
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],input_length=max_sequence_length,trainable=True)
    # define sequence input
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    # from texts_to_sequence to texts_to_sequence embeddings
    embedded_sequences = embedding_layer(sequence_input)

    l_conv1 = Conv1D(128, 5, activation = "relu")(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_conv1)
    l_pool1 = Dropout(0.5)(l_pool1)
    l_conv2 = Conv1D(128, 5, activation = "relu")(l_pool1)
    l_pool2 = MaxPooling1D(5)(l_conv2)
    l_pool2 = Dropout(0.5)(l_pool2)
    l_conv3 = Conv1D(128, 5, activation = "relu")(l_pool2)
    l_pool3 = MaxPooling1D(5)(l_conv3)
    l_pool3 = Dropout(0.5)(l_pool3)
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation = "relu")(l_flat)
    preds = Dense(y_train.shape[1], activation = "softmax")(l_dense)
    model = Model(sequence_input, preds)
    model.compile(loss = "categorical_crossentropy", optimizer = "rmsprop", metrics = ['acc'])
    model.summary()

    # Early stopping
    callbacks = [EarlyStopping(monitor = "val_acc", patience = 50), 
                 ModelCheckpoint(save_dir + var + ".h5", save_best_only = True, monitor = "val_acc")]
    
    return model, callbacks

# create cnn model for a particular variable
def create_cnn_model(save_dir, inter_data, inter_label, var):
    """
    creates a CNN model for a particular variable
    input:
        save_dir -> directory to save model file
        inter_data -> data file containing preprocessed texts_to_sequences (see prep_data in baseline directory if not generated)
        inter_label -> labels created for training and validation examples
        var -> edss_19 or a subscore
    output:
    save file: var.h5 CNN model
    """
    # params
    MAX_SEQUENCE_LENGTH = 1000
    EMBEDDING_DIM = 200
    BATCH_SIZE = 1000
    EPOCHS = 1000

    # Change to intermediate data directory
    os.chdir(inter_data)

    # Load X_train, X_val
    npzfile = np.load("training_data_neurology.npz")
    X_train = npzfile['arr_0']
    X_val = npzfile['arr_1']
    X_test = npzfile['arr_2']

    os.chdir(inter_label)

    # Load y_train, y_val
    npzfile = np.load(var + ".npz")
    y_train = npzfile['arr_0']
    y_val = npzfile['arr_1']
    y_test = npzfile['arr_2']

    # load class weights
    with open(var + "_class_weights.pickle", "rb") as handle:
        class_weights = pickle.load(handle)


    os.chdir(inter_data)
    # Load word embeddings
    with open("ms_word2vec.pickle", "rb") as handle: #
        embeddings_index = pickle.load(handle)

    # Load word_index
    with open("word_index_neurology.pickle", "rb") as handle:
        word_index = pickle.load(handle)

    # define model archetecture
    model, callbacks = model_cnn(y_train, var, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, word_index, embeddings_index, save_dir)

    hist = model.fit(X_train, y_train, 
        validation_data = (X_val,y_val),
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        class_weight=class_weights)

    return
