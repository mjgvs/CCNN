# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:17:23 2018

@author: Maarten
"""

""" This module will contains the functions required for:
    
    1) simulate sum of sigmoids data and store to disk
    2) simulate radial model data and store to disk
    3) train CCNN on sigmoid data and store results to disk
    4) train CCNN on radial data and store results on disk
    5) cnstruct loss, accuracy plots for the CCNN on radial training
    6) construct loss, accuracy plots for the CCNN on sigmoid training
"""

##############################################################################
# 1) SIMULATE SUM OF SIGMOIDS DATA AND STORE ON DISK
###############################################################################
# project-specific modules
import functions_simulate_data
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
""" SUM OF SIGMOIDS DATA"""
# Simulate data
N, TRAIN_SPLIT, SIGNAL_NOISE_RATIO = (10000, 0.5, 2)

(X_train, X_test, 
 Y_train, Y_test,
 Y_train_bin, Y_test_bin) = functions_simulate_data.create_sigmoidsum_data(N=N, 
                                                        train_split=TRAIN_SPLIT, 
                                                        signal_noise_ratio=SIGNAL_NOISE_RATIO,
                                                        seed=1234)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_total = np.vstack((X_train, X_test))
Y_total = np.vstack((Y_train_bin, Y_test_bin))

INPUT_FILE = 'data/SoS_X.pkl'
OUTPUT_FILE = "data/SoS.features"
LABEL_FILE = "data/SoS_Y_bin.pkl"
pkl.dump(X_total, open(INPUT_FILE, 'wb'))
pkl.dump(Y_total, open(LABEL_FILE, "wb"))

##############################################################################
# 2) SIMULATE RADIAL DATA AND STORE ON DISK
###############################################################################
import functions_simulate_data
import pickle as pkl
import numpy as np
""" SUM OF SIGMOIDS DATA"""
# Simulate data
N, TRAIN_SPLIT, SIGNAL_NOISE_RATIO = (10000, 0.5, 2)

(X_train, X_test, 
 Y_train, Y_test,
 Y_train_bin, Y_test_bin) = functions_simulate_data.create_radial_data(N=N,
                        p=10,
                                                        train_split=TRAIN_SPLIT, 
                                                        signal_noise_ratio=SIGNAL_NOISE_RATIO,
                                                        seed=1234)
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

X_total = np.vstack((X_train, X_test))
Y_total = np.vstack((Y_train_bin, Y_test_bin))

INPUT_FILE = 'data/radial_X.pkl'
OUTPUT_FILE = "data/radial.features"
LABEL_FILE = "data/radial_Y_bin.pkl"
pkl.dump(X_total, open(INPUT_FILE, 'wb'))
pkl.dump(Y_total, open(LABEL_FILE, "wb"))


###############################################################################
# TRAIN CCNN ON SIMULATED DATA AND STORE RESULTS TO DISK
###############################################################################
import functions_CCNN
import pickle as pkl
import numpy as np
###############################################################################
# 3) SUMS OF SIGMOID
###############################################################################
N_ITER = 200
N_TRAIN = 5000
LEARNING_RATE = 0.1
PRINT_ITERATIONS = 1

INPUT_FILE = 'data/SoS_X.pkl'
OUTPUT_FILE = "data/SoS.features"
LABEL_FILE = "data/SoS_Y_bin.pkl"

loss_df = {}
train_acc = {}
test_acc = {}
for m in np.array((25, 5, 2, 1)):
    for r in np.array((10, 1, 0.1, 0.01)):
        model = functions_CCNN.ccnn(input_file=INPUT_FILE, 
                            output_file=OUTPUT_FILE, 
                            label_file=LABEL_FILE, 
                            n_iter=N_ITER, 
                            n_train=N_TRAIN, 
                            learning_rate=LEARNING_RATE, 
                            nystrom_dim=m,
                            R=r,
                            print_iterations=PRINT_ITERATIONS)
        model.construct_Q()    
        model.train()
        params = "m"+str(m)+"r"+str(r)
        loss_df[params] = model.train_history[:,0]
        train_acc[params] = model.train_history[:,1]
        test_acc[params] = model.train_history[:,2]
        
# Print validation accuracies for each combination
for key in test_acc:
    print(key, np.max(test_acc[key]))

# Save results for the radial data:
import pickle as pkl
pkl.dump(loss_df, open("results/SoS_ccnn_loss.pkl", "wb"))
pkl.dump(train_acc, open("results/SoS_ccnn_trainacc.pkl", "wb"))
pkl.dump(test_acc, open("results/SoS_ccnn_testacc.pkl", "wb"))

###############################################################################
# TRAIN CCNN ON SIMULATED DATA AND STORE RESULTS TO DISK
###############################################################################
###############################################################################
# 4) RADIAL
###############################################################################
import functions_CCNN
import pickle as pkl
import numpy as np

INPUT_FILE = 'data/radial_X.pkl'
OUTPUT_FILE = "data/radial.features"
LABEL_FILE = "data/radial_Y_bin.pkl"

N_ITER = 200
N_TRAIN = 5000
LEARNING_RATE = 0.1
PRINT_ITERATIONS = 1

loss_df = {}
train_acc = {}
test_acc = {}
for m in np.array((25, 5, 2, 1)):
    for r in np.array((10, 1, 0.1, 0.01)):
        model = functions_CCNN.ccnn(input_file=INPUT_FILE, 
                            output_file=OUTPUT_FILE, 
                            label_file=LABEL_FILE, 
                            n_iter=N_ITER, 
                            n_train=N_TRAIN, 
                            learning_rate=LEARNING_RATE, 
                            nystrom_dim=m,
                            R=r,
                            print_iterations=PRINT_ITERATIONS)
        model.construct_Q()    
        model.train()
        params = "m"+str(m)+"r"+str(r)
        loss_df[params] = model.train_history[:,0]
        train_acc[params] = model.train_history[:,1]
        test_acc[params] = model.train_history[:,2]
        
# Print validation accuracies for each combination
for key in test_acc:
    print(key, np.max(test_acc[key]))

# Save results for the radial data:
import pickle as pkl
pkl.dump(loss_df, open("results/radial_ccnn_loss.pkl", "wb"))
pkl.dump(train_acc, open("results/radial_ccnn_trainacc.pkl", "wb"))
pkl.dump(test_acc, open("results/radial_ccnn_testacc.pkl", "wb"))

###############################################################################
# 5) CONSTRUCT RADIAL CCNN LOSS AND ACCURACY PLOTS AND STORE ON DISK
###############################################################################
import matplotlib.pyplot as plt
import pickle as pkl
# Load results
loss_df = pkl.load(open("results/radial_ccnn_loss.pkl", "rb"))
train_acc = pkl.load(open("results/radial_ccnn_trainacc.pkl", "rb"))
test_acc = pkl.load(open("results/radial_ccnn_testacc.pkl", "rb"))
keys = sorted(loss_df)

# LOSS PLOTS
# m=1
select = keys[0:4]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), loss_df[model])
plt.title('Radial data - CCNN loss, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6935))
plt.savefig('figures/Radial_CCNN_loss_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), loss_df[model])
plt.title('Radial data - CCNN loss, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6935))
plt.savefig('figures/Radial_CCNN_loss_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), loss_df[model])
plt.title('Radial data - CCNN loss, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6935))
plt.savefig('figures/Radial_CCNN_loss_m2.png')
plt.show()    
# m=2
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), loss_df[model])
plt.title('Radial data - CCNN loss, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6935))
plt.savefig('figures/Radial_CCNN_loss_m5.png')
plt.show()   

keys = sorted(train_acc)
# TRAINING ACCURACY PLOTS
import matplotlib.pyplot as plt
import numpy as np
# m=1
select = keys[0:4]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), train_acc[model])
plt.title('Radial data - CCNN training accuracy, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_trainacc_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), train_acc[model])
plt.title('Radial data - CCNN training accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_trainacc_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), train_acc[model])
plt.title('Radial data - CCNN training accuracy, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_trainacc_m2.png')
plt.show()    
# m=2
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), train_acc[model])
plt.title('Radial data - CCNN training accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_trainacc_m5.png')
plt.show() 

keys = sorted(test_acc)
# validation ACCURACY PLOTS
import matplotlib.pyplot as plt
import numpy as np
# m=1
select = keys[0:4]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), test_acc[model])
plt.title('Radial data - CCNN validation accuracy, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_testacc_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), test_acc[model])
plt.title('Radial data - CCNN validation accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_testacc_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), test_acc[model])
plt.title('Radial data - CCNN validation accuracy, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_testacc_m2.png')
plt.show()    
# m=2
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, N_ITER/PRINT_ITERATIONS), test_acc[model])
plt.title('Radial data - CCNN validation accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.4, 0.85))
plt.savefig('figures/Radial_CCNN_testacc_m5.png')
plt.show() 


###############################################################################
# CONSTRUCT LOSS AND ACCURACY PLOTS AND STORE ON DISK
###############################################################################
###############################################################################
# 6) SUMS OF SIGMOID
###############################################################################
import matplotlib.pyplot as plt
import pickle as pkl
# Load results from disk
loss_df = pkl.load(open("results/SoS_ccnn_loss.pkl", "rb"))
train_acc = pkl.load(open("results/SoS_ccnn_trainacc.pkl", "rb"))
test_acc = pkl.load(open("results/SoS_ccnn_testacc.pkl", "rb"))
keys = sorted(loss_df)

# LOSS PLOTS
import matplotlib.pyplot as plt
import numpy as np
# m=1
select = keys[0:4]
for model in select:
    plt.plot(np.arange(0, N_ITER), loss_df[model])
plt.title('Sum-of-sigmoids data - CCNN loss, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.6895, 0.6950))
plt.savefig('figures/SoS_CCNN_loss_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, 200), loss_df[model])
plt.title('Sum-of-sigmoids data - CCNN loss, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.6895, 0.6950))
plt.savefig('figures/SoS_CCNN_loss_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, 200), loss_df[model])
plt.title('Sum-of-sigmoids data - CCNN loss, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6950))
plt.savefig('figures/SoS_CCNN_loss_m2.png')
plt.show()    
# m=5
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, 200), loss_df[model])
plt.title('Sum-of-sigmoids data - CCNN loss, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='upper right')
plt.ylim((0.6895, 0.6950))
plt.savefig('figures/SoS_CCNN_loss_m5.png')
plt.show()   

# TRAINING ACCURACY PLOTS
import matplotlib.pyplot as plt
import numpy as np
# m=1
select = keys[0:4]
showiterations = 50
for model in select:
    plt.plot(np.arange(0, showiterations), train_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN training accuracy, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_trainacc_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, showiterations), train_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN training accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_trainacc_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, showiterations), train_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN training accuracy, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_trainacc_m2.png')
plt.show()    
# m=5
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, showiterations), train_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN training accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_trainacc_m5.png')
plt.show() 

keys = sorted(test_acc)
# TEST ACCURACY PLOTS
import matplotlib.pyplot as plt
import numpy as np
showiterations = 50
# m=1
select = keys[0:4]
for model in select:
    plt.plot(np.arange(0, showiterations), test_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN validation accuracy, m=1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_testacc_m1.png')
plt.show() 
# m=25
select = keys[4:8]
for model in select:
    plt.plot(np.arange(0, showiterations), test_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN validation accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_testacc_m25.png')
plt.show()   
# m=2
select = keys[8:12]
for model in select:
    plt.plot(np.arange(0, showiterations), test_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN validation accuracy, m=2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_testacc_m2.png')
plt.show()    
# m=2
select = keys[12:16]
for model in select:
    plt.plot(np.arange(0, showiterations), test_acc[model][0:showiterations])
plt.title('Sum-of-sigmoids data - CCNN validation accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(select, loc='lower right')
plt.ylim((0.1, 0.9))
plt.savefig('figures/SoS_CCNN_testacc_m5.png')
plt.show()