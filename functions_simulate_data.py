# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:12:37 2018

@author: Maarten
"""

""" This module contains functions to:
    1) Simulate sum-of-sigmoid model data
    2) Simulate radial model data
"""
# public libraries
import numpy as np
import pandas as pd


# helper functions
def tprint(s):
    """
    Enhanced print function with time added to the output.
    """
    import time, sys
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()

###############################################################################
# 1) Simulate sum-of-sigmoid model data
###############################################################################
# project-specific functions
def create_sigmoidsum_data(N, train_split, 
                           signal_noise_ratio=2, 
                           seed=1234, 
                           verbose=True):
    """
    Simulates data using the sum-of-sigmoids structure, as described by [1].    
    Sources:
        [1] Friedman J, Hastie T, Tibshirani R. The elements of statistical 
            learning. New York: Springer series in statistics; 2001.
    """
    # helper functions
    def sigmoid(x):
        return(1/(1+np.exp(-x)))
    def sigmoid_y(X):
        a1 = np.array((3,3)).reshape(2,1)
        a2 = np.array((3,-3)).reshape(2,1)
        s1 = sigmoid(np.dot(a1.T, X.T))
        s2 = sigmoid(np.dot(a2.T, X.T))
        return(s1+s2)
    
    np.random.seed(seed=seed)    
    p=2 # 2 dimensions hardcoded for now
    X = pd.DataFrame(np.random.rand(N, p))
    Y = X.apply(sigmoid_y, axis=1).loc[:,0] # apply sigmoid_y on all samples
    
    # signal-to-noise ratio:
    error_sigma = np.sqrt(np.var(Y))/np.sqrt(signal_noise_ratio) 
    errors = np.random.randn(N) * error_sigma
    Y_obs = Y + errors
    Y_obs = np.array(Y_obs).reshape(N, 1)
    X = np.array(X)
    X_train = X[0:np.int(N*train_split), :]
    X_test = X[np.int(N*train_split):N, :]
    Y_train = Y_obs[0:np.int(N*train_split)]
    Y_test = Y_obs[np.int(N*train_split):N]
    truemean = np.mean(Y.reshape(N,1)[0:np.int(N*train_split)])
    Y_train_bin = (Y_train > truemean)*1
    Y_test_bin = (Y_test > truemean)*1
    
    if verbose==True:
        tprint("Simulated data: sum of sigmoids model")
        tprint("Simulated "+str(N)+" datapoints, of which "+\
               str(N*train_split)+" test points")
    return(X_train, X_test, Y_train, Y_test, Y_train_bin, Y_test_bin)

###############################################################################
# 1) Simulate radial model data
###############################################################################
def create_radial_data(N, p, train_split, 
                       signal_noise_ratio=4, 
                       seed=1234, verbose=True):
    """
    Simulates data using the radial function structure, as described by [1].    
    Sources:
        [1] Friedman J, Hastie T, Tibshirani R. The elements of statistical 
            learning. New York: Springer series in statistics; 2001.
    """
    def radial(x):
        t = x**2
        psi = np.sqrt((1/(2*np.pi))) * np.exp(-t/2)
        return(psi)

    def radial_y(X):
        return(np.prod(radial(X)))
    
    np.random.seed(seed)
    X = pd.DataFrame(np.random.rand(N, p))
    Y = X.apply(radial_y, axis=1)
    # signal-to-noise ratio:
    error_sigma = np.sqrt(np.var(Y))/np.sqrt(signal_noise_ratio) 
    errors = np.random.randn(N) * error_sigma
    Y_obs = Y + errors
    Y_obs = Y_obs.reshape(N, 1)
    X = np.array(X)
    X_train = X[0:np.int(N*train_split), :]
    X_test = X[np.int(N*train_split):N, :]
    Y_train = Y_obs[0:np.int(N*train_split)]
    Y_test = Y_obs[np.int(N*train_split):N]
    truemean = np.mean(Y.reshape(N,1)[0:np.int(N*train_split)])
    Y_train_bin = (Y_train > truemean)*1
    Y_test_bin = (Y_test > truemean)*1
    
    if verbose==True:
        tprint("Simulated data: Radial function model")
        tprint("Simulated "+str(N)+" datapoints, of which "+\
               str(N*train_split)+" test points")
    return(X_train, X_test, Y_train, Y_test, Y_train_bin, Y_test_bin)














