# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:54:40 2018

@author: Maarten
"""

""" This module contains functions to:
    1) Helper functions
    2) Set up a fully connected neural network class 

"""

# public libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

###############################################################################
# 1) Helper functions
###############################################################################
def tprint(s):
    """
    Enhanced print function with time added to the output.
    """
    import time, sys
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()
    
def TestError(Y, Yhat):
    """ Used in NEuralNet_binary.fit_model()
    Calculates the prediction error
    """
    pred_error = (Y - Yhat)**2
    return(np.average(pred_error))


###############################################################################
# 2) Set up a Fully-connected neural network (non-convexified!)
###############################################################################
# classes
class NeuralNet_binary:
    """
    Set up to fit a very simple fully-connected neural network suited for
    classification tasks. Because of the classification tasks it performs, it 
    will train binary cross-entropy as loss function, measure accuracy by zero-
    one loss and optimize using adam.
    """
    def __init__(self):
        pass
    
    def setup_model(self, nr_hidden, input_dim):
        # create model
        self.nr_hidden = nr_hidden
        self.model = Sequential()
        self.model.add(Dense(self.nr_hidden, 
                             input_dim=input_dim, 
                             activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()
        # Compile model
        self.model.compile(loss='binary_crossentropy', 
                           optimizer='adam',
                           metrics=['accuracy'])

    def fit_model(self, 
                  X_train, X_test, Y_train, Y_test, 
                  batch_size=10, epochs=128):
        self.batch_size = batch_size
        self.epochs = epochs
        tprint("Fitting fully-connected Neural Net with "+str(self.nr_hidden)+
               " hidden nodes, for "+str(self.epochs)+"epochs...")
        self.history = self.model.fit(X_train, Y_train, 
                                      epochs=self.epochs, 
                                      batch_size=self.batch_size, 
                                      validation_data=(X_test, Y_test),
                                      verbose=1)
        
        self.Yhat = self.model.predict(X_test)
        self.test_error = TestError(Y=Y_test, Yhat=self.Yhat)
        

