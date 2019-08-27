# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:37:14 2018

@author: Maarten
"""

""" This module contains the functions required to:
    1) Load, tokenize and vectorize clickbait dataset and store on disk
    2) Train a CCNN on the vectorized clickbait data and store loss, acc. on disk
    3) Make loss and accuracy plots.

"""
# project-specific modules
# Public libraries
import pickle as pkl
import numpy as np
# Load project-specific modules
import functions_vectorizeClickbait # used in 1)
import functions_CCNN # used in 2)

###############################################################################
# 1) LOAD, TOKENIZE AND VECTORIZE CLICKBAIT DATASET AND STORE ON DISK
###############################################################################
""" Uses functions from 'functions_vectorizeClickbait.py'. 

We are restricting our vocabulary to the 40000 most observed tokens. Non-
observed tokens will be embedded as all zeroes. Each sequence is padded (or
shortened) to a max length of 20. We are using 50-dimensional pre-trained word
embeddings from GloVe. See functions_vectorizeClickbait.py for more details.
"""
MAX_NB_WORDS=40000
MAX_SEQUENCE_LENGTH=20
EMBEDDING_DIM=50

# Load the raw data:
X_train_raw, Y_train, X_test_raw, Y_test = functions_vectorizeClickbait.load_data()

# Now tokenize the data and pad them to max length of 20 tokens. Also, return
# the tokenizer (a keras.proprocessing.text.Tokenizer object)
X_train, Y_train, X_test, Y_test, tokenizer = \
    functions_vectorizeClickbait.tokenize(X_train_raw, 
                               Y_train, 
                               X_test_raw, 
                               Y_test,
                               max_nb_words=MAX_NB_WORDS,
                               max_sequence_length=MAX_SEQUENCE_LENGTH)

# construct an embedding matrix from the GloVe vectors and a word_index dict.
word_index, embedding_matrix = \
functions_vectorizeClickbait.prepare_embedding(tokenizer=tokenizer,
                                    embedding_dim=EMBEDDING_DIM)

# Leverage the embedding matrix to vectorize our data
X_train = functions_vectorizeClickbait.vectorize_tokens(X_train, 
                                       embedding_matrix=embedding_matrix,
                                       embedding_dim=EMBEDDING_DIM,
                                       max_sequence_length=MAX_SEQUENCE_LENGTH)

X_train = X_train.reshape(X_train.shape[0], MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

X_test = functions_vectorizeClickbait.vectorize_tokens(X_test, 
                                      embedding_matrix=embedding_matrix,
                                      embedding_dim=EMBEDDING_DIM,
                                      max_sequence_length=MAX_SEQUENCE_LENGTH)

X_test = X_test.reshape(X_test.shape[0], MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

# Save our vectorized data as a pkl object so we don't have to re-run code above
INPUT_FILE = 'data/clickbait_data.pkl'
LABEL_FILE = "data/clickbait_labels.pkl"
pkl.dump(np.vstack((X_test, X_train)), open(INPUT_FILE, "wb"))
pkl.dump(np.vstack((Y_test, Y_train)), open(LABEL_FILE, "wb"))

###############################################################################
# 2) TRAIN CCNN ON CLICKBAIT DATA AND STORE RESULTS TO DISK
###############################################################################
""" Uses functions from 'functions_CCNN.py'.

We are training 16 CCNNs on the clickbait data (which is constructed in step 1).
We train for the combinations for Nystroem dimensions (5, 25, 100, 200) and
nuclear norm bound for the parameter matrix ||A||_{*} = (.1, 1, 5, 100). 

We use 750 training iterations for the Projected Stochastic Gradient Descent 
step and half of the data (16000) is used for training. We use a learning rate
(eta) of 0.1 at each step and print results to console at each iteration.
"""


N_ITER = 750
N_TRAIN = 16000 # half of the total set
LEARNING_RATE = 0.1
PRINT_ITERATIONS = 1
INPUT_FILE = 'data/clickbait_data.pkl'
LABEL_FILE = "data/clickbait_labels.pkl"

loss_df = {}
train_acc = {}
test_acc = {}
for m in np.array((200, 100, 25, 5)):
    for r in np.array((100, 5, 1, 0.1)):
        model = functions_CCNN.ccnn(input_file=INPUT_FILE,
                            label_file=LABEL_FILE, 
                            n_iter=N_ITER, 
                            n_train=N_TRAIN,
                            learning_rate=LEARNING_RATE, 
                            nystrom_dim=m,
                            R=r,
                            gamma=0.1,
                            print_iterations=PRINT_ITERATIONS)
        model.construct_Q()    
        model.train() 
        params = "m"+str(m)+"r"+str(r)
        loss_df[params] = model.train_history[:,0]
        train_acc[params] = model.train_history[:,1]
        test_acc[params] = model.train_history[:,2]
        
# Phew, that took a while!

# Print validation accuracies for each combination
for key in test_acc:
    print(key, np.max(test_acc[key]))

# Save results for clickbait data
import pickle as pkl
pkl.dump(loss_df, open("results/clickbait_ccnn_loss.pkl", "wb"))
pkl.dump(train_acc, open("results/clickbait_ccnn_trainacc.pkl", "wb"))
pkl.dump(test_acc, open("results/clickbait_ccnn_testacc.pkl", "wb"))


###############################################################################
# CLICKBAIT CCNN: CONSTRUCT LOSS AND ACCURACY PLOTS AND STORE ON DISK
###############################################################################
""" We make loss, training and validation accuracy plots for the 16 CCNNs 
trained in Step 2) of this module. We set the Y-axis to be equal among all plots
to make comparisons more easier by eye.

"""
N_ITER, PRINT_ITERATIONS = ((750, 1))
loss_df = pkl.load(open("results/clickbait_ccnn_loss.pkl", "rb"))
train_acc = pkl.load(open("results/clickbait_ccnn_trainacc.pkl", "rb"))
test_acc = pkl.load(open("results/clickbait_ccnn_testacc.pkl", "rb"))
keys = sorted(loss_df)

# LOSS PLOTS
import matplotlib.pyplot as plt
import numpy as np
nr_iterations = N_ITER/PRINT_ITERATIONS
# m=200
keys = (('m200r0.1', 'm200r1.0', 'm200r5.0', 'm200r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), loss_df[model])
plt.title('Clickbait data - CCNN loss, m=200')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower left')
plt.ylim((0.682, 0.694))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_loss_m200.png')
plt.show() 
# m=100
keys = (('m100r0.1', 'm100r1.0', 'm100r5.0', 'm100r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), loss_df[model])
plt.title('Clickbait data - CCNN loss, m=100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower left')
plt.ylim((0.682, 0.694))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_loss_m100.png')
plt.show()   
# m=25
keys = (('m25r0.1', 'm25r1.0', 'm25r5.0', 'm25r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), loss_df[model])
plt.title('Clickbait data - CCNN loss, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower left')
plt.ylim((0.682, 0.694))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_loss_m25.png')
plt.show()   
# m=5
keys = (('m5r0.1', 'm5r1.0', 'm5r5.0', 'm5r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), loss_df[model])
plt.title('Clickbait data - CCNN loss, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower left')
plt.ylim((0.682, 0.694))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_loss_m5.png')
plt.show()  

# TRAINING ACCURACY
# m=200
keys = (('m200r0.1', 'm200r1.0', 'm200r5.0', 'm200r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), train_acc[model])
plt.title('Clickbait data - CCNN Training accuracy, m=200')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_trainacc_m200.png')
plt.show() 
# m=100
keys = (('m100r0.1', 'm100r1.0', 'm100r5.0', 'm100r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), train_acc[model])
plt.title('Clickbait data - CCNN Training accuracy, m=100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_trainacc_m100.png')
plt.show()   
# m=25
keys = (('m25r0.1', 'm25r1.0', 'm25r5.0', 'm25r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), train_acc[model])
plt.title('Clickbait data - CCNN Training accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_trainacc_m25.png')
plt.show()   
# m=5
keys = (('m5r0.1', 'm5r1.0', 'm5r5.0', 'm5r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), train_acc[model])
plt.title('Clickbait data - CCNN Training accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_trainacc_m5.png')
plt.show()

# VALIDATION ACCURACY
# m=200
keys = (('m200r0.1', 'm200r1.0', 'm200r5.0', 'm200r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), test_acc[model])
plt.title('Clickbait data - CCNN Validation accuracy, m=200')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_testacc_m200.png')
plt.show() 
# m=100
keys = (('m100r0.1', 'm100r1.0', 'm100r5.0', 'm100r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), test_acc[model])
plt.title('Clickbait data - CCNN Validation accuracy, m=100')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_testacc_m100.png')
plt.show()   
# m=25
keys = (('m25r0.1', 'm25r1.0', 'm25r5.0', 'm25r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), test_acc[model])
plt.title('Clickbait data - CCNN Validation accuracy, m=25')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_testacc_m25.png')
plt.show()   
# m=5
keys = (('m5r0.1', 'm5r1.0', 'm5r5.0', 'm5r100.0'))
for model in keys:
    plt.plot(np.arange(0, nr_iterations), test_acc[model])
plt.title('Clickbait data - CCNN Validation accuracy, m=5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(keys, loc='lower right')
plt.ylim((0.45, 0.95))
plt.xlim((0, N_ITER/PRINT_ITERATIONS))
plt.savefig('figures/clickbait_ccnn_testacc_m5.png')
plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        