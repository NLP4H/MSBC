import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import re
import string
import pickle
# Keras
import tensorflow as tf
print(tf.__version__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dense, Input, Flatten, Concatenate, Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Gensim
from gensim.models import KeyedVectors
# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#sklearn
from sklearn.utils import class_weight



# ML flow
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# --------------------------TRAINING---------------------------------
# Arguments
# GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'


def model_cnn(X_train, y_train, var, embedding_dim, max_sequence_length,
word_index, embeddings_index, class_weights, save_model_dir):
    
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
    print("Missing words: ", count, "Total words: ", embedding_matrix.shape[0])
    print(missing[200:205])
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],input_length=max_sequence_length,trainable=True)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
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
                 ModelCheckpoint(save_model_dir + var + ".h5", save_best_only = True, monitor = "val_acc")]


    return model, callbacks

def model_train(save_model_dir, inter_data, inter_label, var, embedding, max_sequence_length, embedding_dim, batch_size, epochs):
    # params
    MAX_SEQUENCE_LENGTH = max_sequence_length
    EMBEDDING_DIM = embedding_dim
    BATCH_SIZE = batch_size
    EPOCHS = epochs

    # Variable
    var = var

    print("Training variables: ", var)

    # If variables are MRI variables, change data and embedding
    if var in ["mri_worsening", "enhancing_lesion", "num_new_t2_lesions"]:
        embedding = "radiology"

    # Change to intermediate data directory
    #master_path//repo/ML4H_MSProject/data/
    os.chdir(inter_data)
    # Load X_train, X_val
    if embedding == "neurology":
        npzfile = np.load("training_data_neurology.npz")

    if embedding == "radiology":
        npzfile = np.load("training_data_radiology.npz")    
    
    
    X_train = npzfile['arr_0']
    X_val = npzfile['arr_1']
    X_test = npzfile['arr_2']

    os.chdir(inter_label)

    # Load y_train, y_val
    npzfile = np.load(var + ".npz")
    y_train = npzfile['arr_0']
    y_val = npzfile['arr_1']
    y_test = npzfile['arr_2']

    print("Dimensions: ", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

    os.chdir(inter_data)
    if embedding == "neurology":
        # Load word embeddings
        with open("ms_word2vec.pickle", "rb") as handle: #
            embeddings_index = pickle.load(handle)

        # Load word_index
        with open("word_index_neurology.pickle", "rb") as handle:
            word_index = pickle.load(handle)

    if embedding == "radiology":
        # Load word embeddings
        with open("mri_word2vec.pickle", "rb") as handle:
            embeddings_index = pickle.load(handle)

        # Load word_index
        with open("word_index_radiology.pickle", "rb") as handle:
            word_index = pickle.load(handle)

    # Load class weights
    os.chdir(inter_label)
    with open(var + "_class_weights.pickle", "rb") as handle:
        class_weights = pickle.load(handle)

    model, callbacks = model_cnn(X_train, y_train, var, EMBEDDING_DIM, max_sequence_length, word_index, embeddings_index, class_weights, save_model_dir)

    hist = model.fit(X_train, y_train, 
        validation_data = (X_val,y_val),
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        class_weight=class_weights)

    print(hist)

    # evaluate the model
    scores = model.evaluate(X_val, y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))