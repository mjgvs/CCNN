# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 22:38:22 2018

@author: Maarten
"""

"""
This module specifies a number of functions related to loading the clickbait 
dataset and preparing it for use in (convexified and/or convolutional) neural 
networks. 

load_data: Loads the clickbait dataset (input and labels) and splits into 
           train and test sets.
           
tokenize:  Removes punctuation and tokenizes the rest of input data using the
           Keras tokenizer, then pads each input to the max length allowed. 
           
prepare_embedding: Will prepare a matrix containing the token embeddings from
                   the GloVe dataset. The user can choose between 50/100/200/300
                   dimensional token embeddings. Please note, you will need
                   the (large) GloVe files on your disk.
                   
vectorize_tokens:  This will create vectorized dataset by taking the tokenized
                   inputs and using the embedding matrix created by 
                   prepare_embedding to fetch the required vectors.
"""

data_path = 'raw_clickbait_data/'

def load_data(TEST_SPLIT=0.2):
    """This function does the following:
    Load the clickbait datasets
    Select a holdout sample (input: HOLDOUT_SPLIT)
    returns: x_train, y_train, x_test, y_test
    """    
    import numpy as np
        
    # Read in the raw data
    f = open(data_path+'clickbait_data', encoding="utf8") 
    baitlines  =  [line.rstrip().lower() for line in f if len(line.rstrip()) > 0]
    f.close()
    f = open(data_path+'non_clickbait_data', encoding="utf8") 
    nobaitlines  =  [ line.rstrip().lower() for line in f if len(line.rstrip()) > 0]
    f.close()
    headlines = baitlines + nobaitlines
    longest = len(max(headlines, key=len)) # longest headline length
    print("Reading data files...")
    print("Parsed %d bait headlines." % (len(baitlines)))
    print("Parsed %d non bait headlines." % (len(nobaitlines)))
    print("Parsed %d headlines in total." % (len(headlines)))
    print("Longest headline is %d tokens before pre-processing" % longest)
    print("(so including punctuation etc).")
    
    labels = []
    for i in np.arange(len(baitlines)):
        labels.append(1)
    for i in np.arange(len(nobaitlines)):
        labels.append(0)
 
    # get a test set
    from random import shuffle
    # Given list1 and list2
    headlines_shuf = []
    labels_shuf = []
    indices = np.arange(len(headlines))
    shuffle(indices)
    for i in indices:
        headlines_shuf.append(headlines[i])
        labels_shuf.append(labels[i])
    nb_holdout_samples = int(TEST_SPLIT * len(headlines))
    x_train = headlines_shuf[:-nb_holdout_samples]
    y_train = labels_shuf[:-nb_holdout_samples]
    x_test = headlines_shuf[-nb_holdout_samples:]
    y_test = labels_shuf[-nb_holdout_samples:]
    print("Creating a holdout sample...")
    print("Took %d headlines as holdout sample." % nb_holdout_samples)
    print("Remaining %d headlines will be used for training." % len(x_train))

    return(x_train, y_train, x_test, y_test)
    
def tokenize(x_train, y_train, x_test, y_test, 
             max_nb_words=400000, 
             max_sequence_length=26):
    """This function does the following:
        Apply tokenization on headlines
        Apply tokenization on holdout headlines 
        (input for both: MAX_NB_WORDS, MAX_SEQUENCE_LENGTH)
    """
        
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import np_utils
    import numpy as np
        
    print("Initializing Keras tokenizer, max. unique words allowed = %d" % max_nb_words)
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.filters = '#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n'
    tokenizer.fit_on_texts(x_train)
    print("Applying tokenizer on training corpus...")
    x_train = tokenizer.texts_to_sequences(x_train)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    
    print('Padding training sequences to max length %d' % max_sequence_length)
    x_train = pad_sequences(x_train, maxlen=max_sequence_length)
    
    y_train = np_utils.to_categorical(np.asarray(y_train))
    print('Shape of training x_train tensor:', x_train.shape)
    print('Shape of training y_train tensor:', y_train.shape)
    """ Here starts a section dedicated to tokenizing new headlines on which the model has
    not been trained and which may thus contain words that are not in the vocabulary.
    These words will be removed from the new headline and a tokenization will be made
    on the sentence as if it does not contain those words.
    """
    print("Applying tokenizer on holdout corpus...")
    x_test = tokenizer.texts_to_sequences(x_test)
    print('Padding holdout sequences to max length %d' % max_sequence_length)
    x_test = pad_sequences(x_test, maxlen=max_sequence_length)
    y_test = np_utils.to_categorical(np.asarray(y_test))
    print('Shape of holdout x_test tensor:', x_test.shape)
    print('Shape of holdout y_test tensor:', y_test.shape)
    return(x_train, y_train, x_test, y_test, tokenizer)

def prepare_embedding(tokenizer, embedding_dim):
    """This function must accomplish the following:
        set up an embedding_index from pre-trained words (input: EMBEDDING_DIM)
        Set up embedding_matrix for our data
        """
    import numpy as np
        
    """
    Preparing the embedding layer by setting up an embeddings_index. This is a 
    dictionary of 400000 pre-trained word vectors from GLOVE. These come in 
    either 50, 100, 200 or 300 dimensions, depending on how precise you want to
    be and how much memory you are willing to use. 
    """
    print("Initializing word vectors based on %d-dimensional GLOVE dictionary..." % embedding_dim)
    print("... (this could take a minute)")
    word_index = tokenizer.word_index
    
    embeddings_index = {}
    if embedding_dim == 50: f = open(data_path + 'glove.6B.50d.txt', encoding="utf8")
    if embedding_dim == 100: f = open(data_path + 'glove.6B.100d.txt', encoding="utf8")
    if embedding_dim == 200: f = open(data_path + 'glove.6B.200d.txt', encoding="utf8")
    if embedding_dim == 300: f = open(data_path + 'glove.6B.300d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    """
    At this point we can leverage our embedding_index dictionary and our word_index 
    to compute our embedding matrix:
    """
    
    count = 0
    for word, i in word_index.items():
        if word in embeddings_index:
            count += 1
    print("%d out of %d tokens present in GLOVE dictionary" % (count, len(word_index)))
    print("Words not found will be embedded as all-zeros.")
        
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    return(word_index, embedding_matrix)    

def vectorize_tokens(data, embedding_matrix, embedding_dim=50, max_sequence_length=26):
    """
    This function will take as input tokenized data and return a dataframe in
    which the data has been vectorized. 
    
    data: tokenized data that you want to vectorize
    embedding_matrix: matrix where the i'th row contains vector for token 'i'
    embedding_dim: dimension per token vector
    max_sequence_length: max length of an individual input data
    
    clickbait example: tokenized data is provided as an array of shape (N, 26)
    due to truncation after 26 tokens. the 50-dimensional GloVe vectors are used
    for the embedding matrix. The return is an array of shape (N, 50*26). 
    """
    import numpy as np
    
    i = -1
    n = len(data)
    data_vec = np.zeros((n, embedding_dim*max_sequence_length))
    for title in data:
        i += 1
        # vec = np.zeros(EMBEDDING_DIM*MAX_SEQUENCE_LENGTH)
        vec = np.array([])
        for token in title:
            token_vec = embedding_matrix[token]
            vec = np.append(vec, token_vec)
        data_vec[i] = vec
    return(data_vec)