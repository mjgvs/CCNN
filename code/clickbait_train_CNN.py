# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 21:46:24 2018

@author: Maarten
"""

""" This module contains functions to:
    1) Load clickbait data and create word embeddings
    2) Train CNN models on clickbait data and save loss, accuracy plots to disk
    
Notes:
We are using the module 'functions_vectorizeClickbait.py' to create the word-
embedding-version of the raw clickbait strings. See that script for more 
details.
"""

# project-specific modules
import functions_vectorizeClickbait
import functions_CNN_clickbait
MAX_NB_WORDS=40000 # max nr of unique tokens in vocabulary
MAX_SEQUENCE_LENGTH=20 # Nr of tokens. In thesis: P
EMBEDDING_DIM=50 # GloVE embedding dimension. 

###############################################################################
# 1) Load clickbait data and create word embeddings
###############################################################################
# Load the raw data:
X_train, Y_train, X_test, Y_test = \
functions_vectorizeClickbait.load_data(TEST_SPLIT=0.5)

# Now tokenize the data and pad them to max length of 20 tokens. Also, return
# the tokenizer (a keras.proprocessing.text.Tokenizer object)
X_train, Y_train, X_test, Y_test, tokenizer = \
functions_vectorizeClickbait.tokenize(X_train, 
                         Y_train, 
                         X_test, 
                         Y_test,
                         max_nb_words=MAX_NB_WORDS,
                         max_sequence_length=MAX_SEQUENCE_LENGTH)

# construct an embedding matrix from the GloVe vectors and a word_index dict.
word_index, embedding_matrix = \
functions_vectorizeClickbait.prepare_embedding(tokenizer=tokenizer,
                                               embedding_dim=EMBEDDING_DIM)

###############################################################################
# 2) Train CNN models and save loss, accuracy plots to disk
###############################################################################
K_SIZE=1
N_ITERATIONS=40
for N_FILTERS in ((4, 16, 64)):
    model = functions_CNN_clickbait.ConvNeuralNet_binary()
    model.setup_model(embedding_matrix=embedding_matrix,
                      max_sequence_length=MAX_SEQUENCE_LENGTH,
                      n_filters=N_FILTERS,
                      k_size=K_SIZE)
    model.model.summary()
    history = model.fit_model(X_train, X_test, 
                              Y_train, Y_test,
                              batch_size=10, epochs=N_ITERATIONS)    
    import matplotlib.pyplot as plt
    # summarize history for loss
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title('Clickbait data, CNN, ' + str(N_FILTERS) + ' filters, loss')
    plt.ylabel('RMSprop')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim((0.00, 0.26))
    plt.savefig('figures/clickbait_CNN_loss_r'+str(N_FILTERS)+'.png')
    plt.show()
    
    # summarize history for accuracy
    plt.plot(model.history.history['acc'])
    plt.plot(model.history.history['val_acc'])
    plt.title('Clickbait data, CNN, ' + str(N_FILTERS) + ' filters, validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.ylim((0.900, 1.00))
    plt.savefig('figures/clickbait_CNN_testacc_r'+str(N_FILTERS)+'.png')
    plt.show()