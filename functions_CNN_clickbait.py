# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 21:17:49 2018

@author: Maarten
"""

""" This module contains functions to:
    1) Helper functions
    2) Set up a simple CNN network to be applied on the Clickbait dataset
"""

# public libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
import time, sys

###############################################################################
# 1) Helper functions
###############################################################################
def tprint(s):
    """
    Enhanced print function with time added to the output.
    """
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()
    
###############################################################################
# 2) Set up a Fully-connected neural network (non-convexified!)
###############################################################################
class ConvNeuralNet_binary:
    def __init__(self):
        pass
    """ Instantiates a CNN object in Python. 
    setup_model: Set up a simple 1D-convolutional neural network. It uses the 
                embedded sentences as input then runs them through a number of 
                filters, which are summarized using global max pooling. A 
                2-node and softmax layer combination is used to calculate the 
                output. 

    fit_model: Starts training the neural network created using setup_model.    
    """
    def setup_model(self, 
                    embedding_matrix, 
                    max_sequence_length, 
                    n_filters, 
                    k_size):
        # Save important parameters into the class
        self.max_sequence_length = max_sequence_length
        self.n_filters = n_filters
        self.k_size = k_size
        self.vocab_size, self.vocab_dim = embedding_matrix.shape    
        
        # Set up the simple 1D-Convolutional Neural Network 
        tprint("Building a model...")        
        embedding_layer = Embedding(self.vocab_size,
                                    self.vocab_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.max_sequence_length,
                                    trainable=False,
                                    name="Embedding")
        self.model = Sequential()
        self.model.add(embedding_layer)
        self.model.add(Conv1D(filters=self.n_filters, 
                              kernel_size=self.k_size, 
                              activation='relu',
                              strides=1,
                              padding='valid',
                              name="Conv1D"))
        self.model.add(GlobalMaxPooling1D(name="Max_Pooling"))
        self.model.add(Dense(2, activation='softmax', name="Dense_Softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
        
    def fit_model(self, 
                  X_train, X_test, Y_train, Y_test, 
                  batch_size=10, epochs=128):
        self.batch_size = batch_size
        self.epochs = epochs
        tprint("Fitting 1D-Convolutional Neural Net with "+str(self.n_filters)+
               " filters in the hidden layer, for "+str(self.epochs)+"epochs...")
        self.history = self.model.fit(X_train, Y_train, 
                                      epochs=self.epochs, 
                                      batch_size=self.batch_size, 
                                      validation_data=(X_test, Y_test),
                                      verbose=1)
        
###############################################################################
# 3) Classify headlines as (non)clickbait using CNN
###############################################################################
def classify(line, model, tokenizer):
    """ Classify new examples into either clickbait or non-clickbait and return
        a verbose result.
        
        Line: a string
        model: a trained CNN model
        tokenizer: a tokenizer object created during step 1 in 'clickbait_train_CNN.py'
        
        Examples
        --------
        # http://www.bbc.co.uk/bbcthree/article/2b2f79e8-c253-4d1b-9a87-44fe460e5b16
        classify(line="Confession booth: what your bikini waxer is really thinking",
                 model=model,
                 tokenizer=tokenizer)
        
        # https://www.bbc.co.uk/news/business-45055861
        classify(line="Carney: No-deal Brexit risk 'uncomfortably high'",
                 model=model,
                 tokenizer=tokenizer)
        
        # https://www.bbc.co.uk/news/business-45053528
        classify(line="Amazon tax bill falls despite profits leap",
                 model=model,
                 tokenizer=tokenizer)
    """
    import keras
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np
    
    max_sequence_length = model.model.layers[0].input_length
        
    print("Analysing:", line)
    ex = line.lower()
    ex_words_seperated = keras.preprocessing.text.text_to_word_sequence(ex)
    ex_tokenized = np.array(tokenizer.texts_to_sequences(ex_words_seperated)).T
    #There is a hidden error when the word is not present in word_index. Therefore,
    #need to remove unknown words from new input headlines...
    if len(min(ex_tokenized)) == 0: 
        removed = \
        np.asarray(ex_words_seperated)[np.where(np.logical_not(ex_tokenized).astype(int))]
        ex_tokenized = [sum(ex_tokenized, [])]
        print("Had to remove unknown word(s): %s" % removed)
    ex_input = pad_sequences(ex_tokenized, maxlen=max_sequence_length)
    pred = model.model.predict(ex_input)
    pred = np.argmax(pred) # 1 = clickbait, 0 = news
    if pred == 1:
        print("This line is clickbait!" '\n')
    if pred == 0:
        print("This isn't clickbait." '\n')