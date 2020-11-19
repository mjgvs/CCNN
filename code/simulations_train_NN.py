# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 11:56:12 2018

@author: Maarten
"""

""" This module contains functions to:
    1) Simulate Sum of sigmoids data and store scatterplots
    2) Fit fully-connected neural network to sum of sigmoid and store loss, 
       accuracy
    3) Simulate Radial data and store scatterplot
    4) Fit fully-connected neural network to radial and store loss, accuracy
"""

# project-specific modules
import functions_simulate_data
import functions_NN_simulations
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

N = 10000
TRAIN_SPLIT = 0.5
SIGNAL_NOISE_RATIO = 2 # Used to determine variance of noise epsilon
P = 10 # dimension for radial basis function
N_SUBSET = 500 # number of data points to plot in scatterplots

###############################################################################
# 1) SUMS OF SIGMOID simulate data
###############################################################################
""" SUM OF SIGMOIDS MODEL"""
# Simulate data
(X_train, X_test, 
 Y_train, Y_test,
 Y_train_bin, Y_test_bin) = \
 functions_simulate_data.create_sigmoidsum_data(N=N, 
                                                train_split=TRAIN_SPLIT, 
                                                signal_noise_ratio=SIGNAL_NOISE_RATIO,
                                                seed = 1234)

# Plots
# Continuous version
plt.scatter(X_train[0:N_SUBSET,1], X_train[0:N_SUBSET,0], 
            c=Y_train[0:N_SUBSET], cmap = 'gray')
plt.title('Sum of Sigmoids - training data')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.savefig('figures/SoS_scatterplot.png')
plt.show()

# Binary version 
plt.scatter(X_train[0:N_SUBSET,1], X_train[0:N_SUBSET,0], 
            c=Y_train_bin[0:N_SUBSET], cmap = 'gray')
plt.title('Sum of Sigmoids - training data (binarized)')
plt.xlabel('X_1')
plt.ylabel('X_2')
plt.savefig('figures/SoS_scatterplot_binary.png')
plt.show()

###############################################################################
# 2) SUMS OF SIGMOID run neural nets and plot results
###############################################################################
""" SUM OF SIGMOIDS MODEL: BINARY, FITTING NEURAL NETWORK """
EPOCHS = 30 
for NR_HIDDEN in ((5, 25, 100)):
    # shallow fully-connected neural network classifier on binary sum of sigmoids data
    baseline = functions_NN_simulations.NeuralNet_binary()
    baseline.setup_model(input_dim = X_train.shape[1], nr_hidden=NR_HIDDEN)
    baseline.fit_model(X_train=X_train, X_test=X_test, 
                       Y_train=Y_train_bin, Y_test=Y_test_bin,
                       epochs=EPOCHS)
    # summarize history for loss
    plt.plot(baseline.history.history['loss'])
    plt.plot(baseline.history.history['val_loss'])
    plt.title('SumSigmoid - Fully connected NN, ' + str(NR_HIDDEN) + ' nodes, loss')
    plt.ylabel('binary cross-entropy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.ylim((0.35, 0.60))
    plt.savefig('figures/SoS_NN_loss'+str(NR_HIDDEN)+'.png')
    plt.show()    
    # summarize history for accuracy
    np.average(baseline.history.history['val_acc'][-10:])
    plt.plot(baseline.history.history['acc'][0:10])
    plt.plot(baseline.history.history['val_acc'][0:10])
    plt.title('SumSigmoid - Fully connected NN, ' + str(NR_HIDDEN) + ' nodes, accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.ylim((0.79, 0.84))
    plt.savefig('figures/SoS_NN_acc'+str(NR_HIDDEN)+'.png')
    plt.show()

###############################################################################
# 3) radial model simulate data
###############################################################################
""" RADIAL FUNCTION MODEL"""
# simulate data
(X_train, X_test, 
 Y_train, Y_test,
 Y_train_bin, Y_test_bin) = functions_simulate_data.create_radial_data(N=N, 
                        p=P,
                        train_split=TRAIN_SPLIT, 
                        signal_noise_ratio=SIGNAL_NOISE_RATIO,
                        seed=1234)


# plot datapoints in 2 dimensions: continuous labels
from sklearn.decomposition import PCA
mypca = PCA(n_components=2)
X_train_pca = mypca.fit_transform(X_train)
print(mypca.explained_variance_ratio_) 
plt.scatter(X_train_pca[0:N_SUBSET,1], X_train_pca[0:N_SUBSET,0], 
            c=Y_train[0:N_SUBSET], cmap = 'gray')
plt.title('Radial function - training data')
plt.xlabel('PC 1 (10.63%)')
plt.ylabel('PC 2 (10.54%)')
plt.savefig('figures/Radial_scatterplot.png')
plt.show()

# plot datapoints in 2 dimensions: binary labels
plt.scatter(X_train_pca[0:N_SUBSET,1], X_train_pca[0:N_SUBSET,0], 
            c=Y_train_bin[0:N_SUBSET], cmap = 'gray')
plt.title('Radial basis function - training data (binarized)')
plt.xlabel('PC 1 (10.63%)')
plt.ylabel('PC 2 (10.54%)')
plt.savefig('figures/Radial_scatterplot_binary.png')
plt.show()

###############################################################################
# 4) radial data run neural nets and store results
###############################################################################
EPOCHS = 60
""" RADIAL MODEL: BINARY, FITTING NEURAL NETWORK """
for NR_HIDDEN in ((5, 25, 100)):
    # Binary classification version
    baseline = functions_NN_simulations.NeuralNet_binary()
    baseline.setup_model(input_dim = X_train.shape[1], nr_hidden=NR_HIDDEN)
    baseline.fit_model(X_train=X_train, X_test=X_test, 
                       Y_train=Y_train_bin, Y_test=Y_test_bin,
                       epochs=EPOCHS)    
    # summarize history for loss
    plt.plot(baseline.history.history['loss'])
    plt.plot(baseline.history.history['val_loss'])
    plt.title('Radial - Fully connected NN, '+ str(NR_HIDDEN)+' nodes, loss')
    plt.ylabel('binary cross-entropy')
    plt.xlabel('epoch')
    plt.ylim((0.35, 0.8))
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig('figures/Radial_NN_loss'+str(NR_HIDDEN)+'.png')
    plt.show()
    # summarize history for accuracy
    np.average(baseline.history.history['val_acc'][-10:])
    plt.plot(baseline.history.history['acc'])
    plt.plot(baseline.history.history['val_acc'])
    plt.title('Radial - Fully connected NN, '+ str(NR_HIDDEN)+' nodes, accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.ylim((0.45, 0.85))
    plt.savefig('figures/Radial_NN_acc'+str(NR_HIDDEN)+'.png')
    plt.show()

###############################################################################
# Some extra code to check whether the observed and true binary values are equal
# (they are not: there is about 15% false classification introduced due to noise)
def glance(X):
    return(X[1:5])


true_average = np.average(Y)
obs_average = np.average(Y_obs)

obs_average
true_average

glance(Y)
glance(Y_obs)

Y_bin = (Y > true_average)*1
Y_obsbin = (Y_obs > obs_average)*1

Y_bin.shape
Y_obsbin.shape

np.array_equal(Y_bin, Y_obsbin)
np.sum((Y_bin == Y_obsbin)*1)