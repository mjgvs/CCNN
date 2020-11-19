# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:36:33 2018

@author: Maarten
"""

""" This module will contains the functions required for:
    
    1) Helper functions
    2) Projecting a matrix on nuclear norm ball
    3) Projected Gradient Descent of a matrix on the nuclear norm ball
    4) train CCNN on radial data and store results on disk
    5) cnstruct loss, accuracy plots for the CCNN on radial training
    6) construct loss, accuracy plots for the CCNN on sigmoid training
"""

# public libraries
import numpy as np
from numpy import linalg as LA
import sys
import numexpr as ne
from sklearn.preprocessing import label_binarize
import time
import pickle as pkl

###############################################################################
##### 1) Helper functions #####################################################
###############################################################################
# helper functions        
def tprint(s):
    """ Enhanced print function with time added to the output.
    Source: Zhang et al.
    """
    tm_str = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    print(tm_str + ":  " + str(s))
    sys.stdout.flush()    
    
###############################################################################
### 2) ALGORITHM 2: project a matrix on nuclear norm ball #####################
###############################################################################
""" Algorithm 2 in MSc. Thesis of Maarten van Schaik
The code for these three functions is heavily inspired by that of Zhang et al.
in their CCNN script.

The three functions below compute the steps for Algorithm 2.

Main function: project_to_trace_norm, which requires the functions
euclidean_proj_simplex, euclidean_proj_l1ball
"""
    
def project_to_nuclear_norm(A, R, P, nystrom_dim, d2):
    """ Main function for Algorithm 2
    Dependencies: euclidean_proj_simplex, euclidean_proj_l1ball
    
    Parameters
    ----------
    A:  numpy array,
        matrix to be projected onto the nuclear norm ball
    R: int, 
       upper bound of nuclear norm.
    P: int,
       number of patches (on clickbait: sequence length)
    nystroem_dim: int,
                  Nystroem dimension. In Thesis: m
    d2: int,
        Number of classes for the categorical classification
        
    Returns
    -------
    Ahat: numpy array,
          Projection of A on the nuclear norm ||A||_{*} = R
    U, s, V: singular vectors and values of A
    """
    A = A.reshape((d2-1)*P, nystrom_dim)
    # A = np.reshape(A, ((n_classes-1)*P, nystroem_dim))
    (U, s, V) = LA.svd(A, full_matrices=False)
    s = euclidean_proj_l1ball(s, s=R)
    Ahat = np.reshape(np.dot(U, np.dot(np.diag(s), V)), ((d2-1), P*nystrom_dim))
    return Ahat, U, s, V

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        $min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0$
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

def euclidean_proj_l1ball(v, s=1):
    """ Compute the Euclidean projection on a L1-ball
    Solves the optimisation problem (using the algorithm from [1]):
        $min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s$
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the L1-ball
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the L1-ball of radius s
    Notes
    -----
    Solves the problem by a reduction to the positive simplex case
    See also
    --------
    euclidean_proj_simplex
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # compute the vector of absolute values
    u = np.abs(v)
    # check if v is already a solution
    if u.sum() <= s:
        # L1-norm is <= s
        return v
    # v is not already a solution: optimum lies on the boundary (norm == s)
    # project *u* on the simplex
    w = euclidean_proj_simplex(u, s=s)
    # compute the solution to the original problem on v
    w *= np.sign(v)
    return w
                      
################################################################################ 
# 3) ALGORITHM 3: Projected Gradient Descent of a matrix on the nuclear norm ball #
################################################################################ 
def evaluate_classifier(X_train, X_test, Y_train, Y_test, A, d2):
    """ Evaluates quality of the classifier by computing the loss and the 
    classification error in the train and test set. The loss used is categorical
    cross-entropy and the classification error is zero-one loss.    
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    eXAY = np.exp(np.sum((np.dot(X_train, A.T)) * Y_train[:,0:(d2-1)], 
                         axis=1)) 
    eXA_sum = np.sum(np.exp(np.dot(X_train, A.T)), axis=1) + 1
    loss = - np.average(np.log(eXAY/eXA_sum))

    predict_train = np.concatenate((np.dot(X_train, A.T), np.zeros((n_train, 1), 
                                           dtype=np.float32)), axis=1)
    predict_test = np.concatenate((np.dot(X_test, A.T), np.zeros((n_test, 1), 
                                          dtype=np.float32)), axis=1)

    error_train = np.average(np.argmax(predict_train, axis=1) != \
                             Y_train.reshape(n_train,))
    error_test = np.average(np.argmax(predict_test, axis=1) != \
                            Y_test.reshape(n_test,))

    return loss, error_train, error_test

def projected_gradient_descent(X_train, 
                               Y_train, 
                               X_test, 
                               Y_test, 
                               R, 
                               n_iter, 
                               learning_rate,
                               print_iterations):
    """ In order to solve optimization problem in Algorithm 1, projected 
    gradient descent is used. At iteration t, using a step size $\eta > 0$,
    it forms the new matrix $A^{t+1}$ based on the previous iterate $A^t$ 
    according to:
        
        $$A^{t+1} = \Prod_R(A^t - \eta \nabla_A L(A^t)),$$
        
    where $\nabla_A L(A^t)$ denotes the gradient of the objective function
    defined in Algorithm 1,, and $\Prod_R$ denotes the Euclidean projection 
    onto the nuclear norm ball $\{A: \|A\|_{*} \leq R\}$. 
    
    ---------------------------------------------------------------------------
    
    This function takes as input Will return the learned 
    parameter matrix A \in R^{m , P*d_2}.

    X_train, X_test: train and test data (X_reduced[0:n_train] etc) Respective 
                     sizes decided by n_train. Here is $Q \in R^{n, P, m}$, 
                     the low-dimensional feature representation of the 
                     kernelized patches (as returned by the Nystrom transform) 
                     
    Y_train, Y_test: train and test labels.
    
    For clarification:
    P: Number of patches. In the clickbait data, the 20-token sequences are
       evaluated with stride=1, (so each token is its own patch), so P=20.
        
    nystrom_dim: Dimension of the $Q$ matrix. In clickbait data, there is 1 
                 channel and nystrom_dim=m, such that nystrom_dim=1*m=m.
        
    R:   toplevel input by user for. It is the hyperparameter R: the radius of 
         the nuclear norm ball onto which the parameter matrix A is projected;
    """
    d2 = len(np.unique(Y_train))         
    n_train, P, nystrom_dim = X_train.shape
    n_test = X_test.shape[0]
    X_train = X_train.reshape(n_train, P*nystrom_dim)
    X_test = X_test.reshape(n_test, P*nystrom_dim)
    A = np.random.randn((d2-1), P*nystrom_dim) 
    A_sum = np.zeros(((d2-1), P*nystrom_dim), dtype=np.float32)

    # setup for objects to store performance during training
    loss_history = np.array(())
    error_train_history = np.array(())
    error_test_history = np.array(())
    
    # Projected Stochastic Gradient Descent
    mini_batch_size = 50
    nr_of_mini_batches = 10
    for t in range(n_iter):
        # Non-projected Stochastic Gradient Descent
        for i in range(0, nr_of_mini_batches):
            """ This double for loop is largely inspired by Zhang et al's 
            version in their CCNN code.
            
            For each batch in nr_of_mini_batches, randomy select mini_batch_size 
            training samples inputs and labels. For this mini_batch_size samples, 
            calculate the steps for stochastic gradient descent such that:
                
            $\nabla_A{_k} L(A_k) = 
                  frac{1}{n}X(frac{exp(XA_k^T)}{\sum_{k=1}^K exp(XA_k^T)} - Y)$
            
            where A_k are the parameters for class K, X the mini batch of data, 
            Y the targets of the mini batch, and n the mini batch size. L(A_k)
            is the loss function for multiclass classification (softmax)
            
            The update rule is then that $A_{t+1} = 
                                             A_{t} - \gamma \nabla_{A_k}L(A_k)$
            
            After the update is applied, the new coefficients are projected on
            the nuclear norm.
            """
            # randomly sample mini_batch_size (=50) patches
            index = np.random.randint(0, n_train, mini_batch_size) 
            X_sample = X_train[index] # dimensions: (50, P*nystrom_dim)
            # one column removed (because inferred from other columns): 
            Y_sample = Y_train[index, 0:(d2-1)] 

            # stochastic gradient descent
            XA = np.dot(X_sample, A.T) 
            eXA = ne.evaluate("exp(XA)") 
            eXA_sum = np.sum(eXA, axis=1).reshape((mini_batch_size, 1)) + 1 
            diff = ne.evaluate("eXA/eXA_sum - Y_sample")  
            grad_A = np.dot(diff.T, X_sample) / mini_batch_size 
            A -= learning_rate * grad_A # average A after nr_of_mini_batches times
            
        # projection to nuclear norm
        A, U, s, V = project_to_nuclear_norm(A=A, 
                                             R=R,
                                             P=P, 
                                             nystrom_dim=nystrom_dim,
                                             d2=d2)
        A_sum += A
        if (t+1) % print_iterations == 0:
            # percentage of 'variance' in top 25 singular values:
            dim = np.sum(s[0:25]) / np.sum(s) 
            A_avg = A_sum / 250
            loss, error_train, error_test = evaluate_classifier(X_train=X_train,
                                                                X_test=X_test, 
                                                                Y_train=Y_train, 
                                                                Y_test=Y_test, 
                                                                A=A_avg,
                                                                d2=d2)
            loss_history = np.append(loss_history, np.array(loss))
            error_train_history = np.append(error_train_history, 
                                            np.array(error_train))
            error_test_history = np.append(error_test_history, 
                                           np.array(error_test))
            
            A_sum = np.zeros(((d2-1), P*nystrom_dim), dtype=np.float32) # reset A_sum

            tprint("iter " + str(t+1) + 
                   ": loss=" + str(loss) + 
                   ", train accuracy =" + str(error_train) + 
                   ", test accuracy =" + str(error_test))
            
        history = np.concatenate((loss_history[:,np.newaxis], 
                                  error_train_history[:,np.newaxis], 
                                  error_test_history[:,np.newaxis]), 
            axis=1)

    """ Once the final iterations have been made, the final trace-projected 
    coefficients are calculated and returned. 
    """
    A_avg, U, s, V = project_to_nuclear_norm(A=np.reshape(A_avg, ((d2-1)*P, 
                                                                  nystrom_dim)), 
                                             R=R, 
                                             P=P, 
                                             nystrom_dim=nystrom_dim,
                                             d2=d2)
    dim = min(np.sum((s > 0).astype(int)), 25)
    return A_avg, V[0:dim], history

###############################################################################
####################### ALGORITHM 1: LEARN A CCNN #############################
###############################################################################
class ccnn:
    """ Function class to compute Algorithm 1 from MSc. Thesis.
    
    Functions:
        __init__: initializes the ccnn class. 
        construct_Q: From the input data, construct Q by approximating the 
                     kernel matrix K.
        train: Trains the CCNN on Q and Y.
    """
    def __init__(self, 
                 input_file, 
                 label_file, 
                 n_train, 
                 nystrom_dim, 
                 gamma, 
                 R, 
                 learning_rate, 
                 n_iter, 
                 print_iterations):
        """ Initializes the CCNN model with input from user.
  
        Parameters
        ----------
        input_file, label_file: path directories for X and Y.
                                X must be a (N, P, d1) array.
                                Y must be a (N,) array.
        nystrom_dim: Nystroem dimension used to approximate Q during
                     step 1 of Algorithm 1 in MSc. Thesis. In Thesis: "m".
        gamma: hyperparameter for the RBF kernel. 
        R: Nuclear Norm radius to project A on: ||A||_{*} = R
        n_iter: number of iterations for the Projected Stochastic Gradient Descent
        print_iterations: how often to print current loss and accuracy to 
                          the console. 1 means it prints each iteration.
        """
        tprint("read from " + input_file)
        # Storing data
        self.X_raw = pkl.load(open(input_file, "rb"))
        self.label = pkl.load(open(label_file, "rb"))[:, 0]
        # Storing data properties
        self.d2 = np.unique(self.label).shape[0] 
        self.n = self.X_raw.shape[0]
        self.P = self.X_raw.shape[1]
        self.d1 = self.X_raw.shape[2]
        # Storing hyperparameters
        self.n_train = n_train
        self.n_test = self.n - self.n_train
        self.nystrom_dim = nystrom_dim # in Thesis: m
        self.gamma = gamma 
        self.R = R
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.print_iterations = print_iterations
        
        # construct patches
        # This is an artifact from when X_raw was not neccesarily supplied as a
        # 3-dimensional array, and n, P, d1 were given as user input.
        # For sequence data it's easy: just use .reshape
        self.Z = self.X_raw.reshape(self.n, 
                                        self.P, 
                                        self.d1)
        tprint("Data contains " + str(self.n) + " samples, with " + 
               str(self.P) + " patches of dimension " + str(self.d1) + ".") 
        tprint("Output contains " + str(self.n) + " samples, with " + 
               str(self.d2) + " classes.") 
        
    def construct_Q(self, feature_normalization=True):
        """ Step 1 and 2 in Algorithm 1 in MSc. Thesis.
        Computes Q, such that QQ^T \approx K, where K is the RBF kernel matrix.
        Also applies normalization to the features by default. This is carried
        over by example from the CCNN code of Zhang et. al
            
        Input
        Z_train, Z_test: (N,P,d1) arrays. Each Z[i,:,:] is one Z(x_i). 
                         Result from __init__.
        
        Output
        Q_train, Q_test: (N,P,m) arrays Each Q[i,:,:] is one Q(x_i). 
                         Used in train() function below.
        """    
        from sklearn.kernel_approximation import Nystroem
        import numpy as np
        import math
        tprint("Using Scikitlearn Nystroem function")
        tprint("Creating Q...")        
        Z_train = self.Z[0:self.n_train].reshape((self.n_train*self.P, self.d1))
        Z_test = self.Z[self.n_train:self.n].reshape((self.n_test*self.P, self.d1))
        transformer = Nystroem(gamma=self.gamma, n_components=self.nystrom_dim)
        transformer = transformer.fit(X=Z_train)
        Q_train = transformer.transform(Z_train)
        Q_test = transformer.transform(Z_test)
        self.Q_train = Q_train.reshape((self.n_train, self.P, self.nystrom_dim))
        self.Q_test = Q_test.reshape((self.n_test, self.P, self.nystrom_dim))
        
        if feature_normalization==True:
            self.Q_train = self.Q_train.reshape((self.n_train*self.P, 
                                                 self.nystrom_dim))
            self.Q_train -= np.mean(self.Q_train, axis=0)
            self.Q_train /= LA.norm(self.Q_train) / math.sqrt(self.n_train*self.P)
            self.Q_train = self.Q_train.reshape((self.n_train,
                                                 self.P, self.nystrom_dim))
            self.Q_test = self.Q_test.reshape((self.n_test*self.P,
                                                 self.nystrom_dim))
            self.Q_test -= np.mean(self.Q_test, axis=0)
            self.Q_test /= LA.norm(self.Q_test) / math.sqrt(self.n_train*self.P)
            self.Q_test = self.Q_test.reshape((self.n_test,
                                                 self.P, self.nystrom_dim))        
    # Training CCNN
    def train(self): 
        """ Algorithm 1 from MSc. Thesis. 
        Trains the CCNN using Projected Stochastic Gradient Descent. 
        
        It solves the constrained optimization problem of step 3 in Algorithm 1.
        
        Requires:
        -----
        Q_train, Q_test: Output from construct_Q function
        Y_train, Y_test: User input on class level
        
        Label_binarize will create a one-hot encoding matrix with K-1 columns,
        showing a 1 in each row only once to indicate which class the case 
        belongs to.
        
        Parameters
        ----------
        n_iter: number of iterations for the Projected Stochastic Gradient Descent
        print_iterations: how often to print current loss and accuracy to 
                          the console. 1 means it prints each iteration.        
        """        
        tprint("Training CCNN using projected stochastic gradient descent...")
        from sklearn.preprocessing import label_binarize
        binary_label = label_binarize(self.label, classes=range(0, self.d2))
        
        self.Y_train=binary_label[0:self.n_train] 
        self.Y_test=binary_label[self.n_train:] 
                                                         
        self.A, self.filter, self.train_history = \
            projected_gradient_descent(X_train=self.Q_train,
                                       Y_train=self.Y_train, 
                                       X_test=self.Q_test, 
                                       Y_test=self.Y_test,
                                       n_iter=self.n_iter, 
                                       print_iterations=self.print_iterations,
                                       R=self.R, 
                                       learning_rate=self.learning_rate)