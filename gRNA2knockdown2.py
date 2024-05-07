#!/usr/bin/env python3.11
__author__ = "Alec Taylor, Enoch Yeung"
__version__ = "1.0.0"
__maintainer__ = "Alec Taylor"
__email__ = "aztaylor76@fastmail.com"
__status__ = "Development"

import os
import csv
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import platereadertools as pr

from sklearn.preprocessing import MinMaxScaler
import numpy as np

'''This module contains the functions to encode DNA sequences and for use in predicting  the knockdown 
efficiency of CRISPR dCasRx-gRNAs effectors. The model takes in the RNA sequence of the targeted transcript 
embedds the sequence using an autoencoder. Subsequently, the model takes the embedded sequence and the mRNA of the target
to predict the knockdown efficiency. The base design is adaptable with a variable number of hidden layers and units and
is meant to be able to construct autoencoders, feedforward NNs and residual networks.

The model is trained using  the Adam optimizer and a custom VAE embbed loss function to determine the mean squared error.
The original code was developed by Enoch Yeung in the Biological Control Laboratory at the University of California, 
Santa Barbara. Some things to note:

 - The code is written in Python 3.11 and uses the TensorFlow 2.x library.
 - The code is written in a modular fashion and can be used as a module in other scripts.
 - The code is written in a functional programming style and uses type hints to define the types of the arguments and
 outputs.
 - The module contiains a __main__ function that runs the code when the module is run as a standalone script.
 - To run the code as a module, a tensorflow session must be defined and the code must be run in a tensorflow session.
 - Becuase this script uses placeholder variabeles, which are incompatible with the eager execution mode, the eager
    execution mode must be disabled before running the code. This can be done by running the following code:
    tf.compat.v1.disable_eager_execution()
'''

# Define Auxillary Functions
def sequence_encoding(sequence: str) -> np.array:
    '''Encode DNA sequence into a 1D array of floats. The encoding is mapped as to be normalized between 0 and 1.
    args:
        sequence: str, DNA sequence
    returns:
        np.array, 1D array of floats
    '''
    seq = sequence.lower()
    encoding  = []
    encoding_map = {'a':'0.25', 'c':'0.5', 'g':'0.75', 't':'1'}
    for i in range(len(seq)):
        encoding.append(float(encoding_map[seq[i]]))
    return np.asarray(encoding)

def make_labeled_corpus(seq_list, label_list, stride_param):
    '''Create a corpus of sequences and their corresponding labels. The corpus is created by taking a sequence and
    sliding a window of size stride_param over the sequence. The labels are the same for each window.
    args:
        seq_list: list, list of where the first index is the sequence and the second index is the label.
        stride_param: int, stride parameter
    returns:
        corpus: list, list of sequences
        labels: list, list of labels
    '''
    corpus = []
    labels= []
    for i, this_seq in enumerate(seq_list):
        this_seq = this_seq[0]
        this_label = label_list[i]
        for ind in range(0,len(this_seq)-stride_param+1):
            this_datapt = this_seq[ind:ind+stride_param]
            corpus.append(this_datapt)
            labels.append(this_label)
    return corpus,labels

def create_corpus(seq_array:np.ndarray, y_trace: np.ndarray, stride=1) -> dict:
    '''Create a corpus of sequences and their corresponding labels (y_trace).
    args:
        seq_array: np.ndarray, 2D array of 
        y_trace: np.ndarray, 2D array of y traces (OD values or Fluorescence)
    returns:
        labeled_corpus: dict, dict of data and labels (sequence, y_trace)
    '''
    corpus = {}
    if seq_array.shape[0] != y_trace.shape[0]:
        raise ValueError('seq_array and y_trace have different number of rows')
    
    for seq in seq_array:
        for i in range(len(seq)):
            corpus[seq[i]] = y_trace[i]
    return corpus# leave as is in the original code. Optimize Later

def xavier_init(n_inputs: int, n_outputs: int, uniform=True) -> tf.initializers:
    '''Initialize weights with Xavier initialization. From Enoch's code this initializes
    the weights with a uniform distribution to keep the scale if gradients roughly the same in all layers.
    Originally from Xavier Glorot and Yoshua Bengio (2010).
    args:
        n_inputs: int, number of inputs
        n_outputs: int, number of outputs
        uniform: bool, if True use uniform distribution, else use normal distribution
        
    returns:
        tf.initializer, tensorflow initializer
    '''
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def weight_Variable(shape: tuple) -> tf.Variable:
    '''Create a weight Variable with a given shape and name. Defined to be used in the standard definitiion
    of a neuron: Wx + b, where W is the weight, x is the input and b is the bias.
    args:
        shape: tuple, shape of the weight
    returns:
        tf.Variable, tensorflow Variable
    '''
    std_dev = math.sqrt(3.0/(shape[0] + shape[1]))
    return tf.Variable(tf.random.truncated_normal(shape, 
                                                  mean=0.0, 
                                                  stddev=std_dev,
                                                  dtype=tf.float32))

def bias_Variable(shape: tuple) -> tf.Variable:
    '''Create a bias Variable with a given shape and name. Defined to be used in the standard definitiion
    of a neuron: Wx + b, where W is the weight, x is the input and b is the bias.
    args:
        shape: tuple, shape of the bias
    returns:
        tf.Variable, tensorflow Variable
    '''
    print("shape: ".format(shape))
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.random.truncated_normal(shape, mean=0.0,
                                                  stddev=std_dev,
                                                  dtype=tf.float32))

def initialize_Wblist(n_u, hv_list) -> (list, list): # type: ignore
    '''Initialize the weights and biases for the network. The weights are initialized using the weight_Variable function 
    and the biases are initialized using the bias_Variable function. The weights and biases are stored in lists.
    args:
        n_u: int, number of inputs
        hv_list: list, list of hidden layer widths
    returns:
        W_list: list, list of weights
        b_list: list, list of biases
    '''
    W_list = []
    b_list = []
    n_depth = len(hv_list)
    print("Length of hv_list: " + repr(n_depth))
    
    #hv_list[n_depth-1] = n_y;
    for k in range(0,n_depth):
        if k==0:
            W1 = weight_Variable([n_u,hv_list[k]])
            b1 = bias_Variable([hv_list[k]])
            W_list.append(W1)
            b_list.append(b1)
        else:
            W_list.append(weight_Variable([hv_list[k-1],hv_list[k]]))
            b_list.append(bias_Variable([hv_list[k]]))
    return W_list, b_list

def rvs(dim=3):
    '''Generate a random orthogonal matrix. The matrix is generated using the Householder transformation. This should 
    scrabble the 4-hot encoding to project into random input space. This improves performance for reason I do not yet know.
    args:
        dim: int, dimension of the matrix
    returns:
        H: np.array, random orthogonal matrix
    '''
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim-n+1,))
        D[n-1] = np.sign(x[0])
        x[0] -= D[n-1]*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D*H.T).T
    return H

# Define the loss functions
def embed_loss(y_true,embed_true):
    '''Calculate the embedding loss. The embedding loss is the mean squared error between the predicted and true
        embeddings.
        args:
            y_true: tf.Variable, true y values
            embed_true: tf.Variable, true embeddings
        returns:
            tf.Variable, embedding loss
    '''
    #y_true is (batch_size_param) x (dim of stride) tensor
    IP_Matrix_y = tf.matmul(y_true,tf.transpose(y_true))
    IP_Matrix_e = tf.matmul(embed_true,tf.transpose(embed_true))
    Scale_Matrix_y = tf.linalg.tensor_diag(tf.norm(y_true,ord='euclidean'
                                                   ,axis=1))
    Scale_Matrix_e = tf.linalg.tensor_diag(tf.norm(embed_true,ord='euclidean'
                                                   ,axis=1))
    Ky = tf.matmul(tf.matmul(Scale_Matrix_y,IP_Matrix_y),Scale_Matrix_y)
    Ke = tf.matmul(tf.matmul(Scale_Matrix_e,IP_Matrix_e),Scale_Matrix_e)
    return tf.norm(IP_Matrix_y-IP_Matrix_e,axis=[0,1],ord='fro')/tf.norm(
                    IP_Matrix_y,axis=[0,1],ord='fro')

def vae_loss(y_model,y_true):
    '''Calculate the VAE loss. The VAE loss is the mean squared error between the predicted and true y values.
    args:
        y_model: tf.Variable, predicted y values
        y_true: tf.Variable, true y values
    returns:
        tf.Variable, VAE loss
    '''
    return tf.norm(y_true - y_model,axis=[0,1],ord=2)/tf.norm(y_true,axis=[0,1]
                                                              ,ord=2)
    
def customLoss(y_model:tf.Variable, y_true:tf.Variable,
               embed_true:tf.Variable) -> tf.Variable:
    '''Custom loss function that combines the VAE loss and the embedding loss. The VAE loss is the mean squared error
    between the predicted and true y values. The embedding loss is the mean squared error between the predicted and true
    embeddings.
    args:
        y_model: tf.Variable, predicted y values
        y_true: tf.Variable, true y values
        embed_true: tf.Variable, true embeddings
    returns:
        tf.Variable, custom loss
        '''
    return vae_loss(y_model,y_true)+embed_loss(y_true,embed_true)


# Define the network
def create_model(input_shape, embedding_dim, intermediate_dim, label_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(intermediate_dim, activation='elu', input_shape=input_shape),
        tf.keras.layers.Dense(embedding_dim, activation='elu', input_shape=intermediate_dim),
        tf.keras.layers.Dense(intermediate_dim, activation='elu',input_shape= embedding_dim),
        tf.keras.laters.Dense(label_dim, activation='elu', input_shape=intermediate_dim)
        ])
    return model

#Train and test the network
def train_model(model, X_train, y_train, X_valid, y_valid, batch_size,
                max_epochs, loss=customLoss):
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.25),
                  loss=loss)
    history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid), verbose=1)
    return history

SeqMap = ['A','C','T','G']

def elemback2seq(this_elem):
    '''Convert the lattice-like vector representation/encoding to a DNA sequence. The encoding is mapped as to be
        normalized between 0 and 1.
        args:
            this_elem: float, lattice-like vector representation/encoding
        returns:
            str, DNA sequence
    '''
    this_elem = this_elem[0]
    seq_dist_list = list(np.abs(np.array([this_elem]*4) - np.array([0.25,0.5,0.75,1.0])))
    opt_index = seq_dist_list.index(np.min(np.array(seq_dist_list)))
    return(SeqMap[opt_index])

def vecback2seq(untransformed_vec):
    '''Convert the lattice-like vector representation/encoding to a DNA sequence. The encoding is mapped as to be
        normalized between 0 and 1.
        args:
            untransformed_vec: np.array, lattice-like vector representation/encoding
        returns:
            list, list of DNA sequences
    '''
    # untransformed_vec is computed using the inverse of Rand_Transform to recover lattice-like vector representation/encoding
    seq_out = [elemback2seq(elem) for elem in untransformed_vec]
    return seq_out

def num_mismatch(seq_model,seq_true):
    return np.sum(1*([not(seq_model[ind]==seq_true[ind]) for ind in range(0,len(seq_true))]))

# Run if not imported
if __name__ == "__main__":
    '''This is the main function that runs the code. It loads the data, organizes the data, encodes the data,
       initializes the network, trains the network and tests the network. The data is loaded from the Data directory
       and the label data is organized using the platereadertools.py module. The data is then encoded using the
       sequence_encoding function. The network is initialized using the initialize_Wblist function and trained using
       the train_net function. The network is then tested using the test_net function. If not loaded as a standalone
       script, this script will act as a module and the functions can be imported and used in other scripts.
    '''
    
    # First we need to load the data
    data_fp = "Data/"
    spacer_fp = os.path.join(data_fp, "GFP_spacers.gbk")
    data_0nM_fp = os.path.join(data_fp,
                                "p2x11_80memberlibrary_0mMIPTG20230730.txt")
    data_10mM_fp = os.path.join(data_fp,
                                "p2x11_80memberlib_10mMIPTG20230730.txt")

    # Organize the label, sequence data from platereadertools and the csv standard module.
    seqs = csv.reader(open("Data/GFP_spacers.csv"))
    allseqs = [seq for seq in seqs]
    data0, time0 = pr.Organize(data_0nM_fp, 8, 12, 18, 3/60)
    data1, time1 = pr.Organize(data_10mM_fp, 8, 12, 18, 3/60)

    # Based off of the timeseries data, we can see that the greatest change in flourescence occurs at timepoint 165 
    # (~8hours). We will use this timepoint to calculate the fold change between the 0mM and 10mM data.
    reads = list(data0.keys())
    data_pt0 = data0[reads[1]][:,:,165]
    data_pt1 = data1[reads[1]][:,:,165]

    # Calculate the fold change between the 0mM and 10mM data.
    fold_change = data_pt1/data_pt0
    data = np.reshape(fold_change,(96))

    # Visualize the foldchange to see if there are any trends.
    explore = False
    if explore == True:
        fig, ax = plt.subplots(1,1)
        ax.bar(range(len(data)), data)
        ax.set_xlabel("gRNA position with tilling equal to 3bp")
        ax.set_ylabel('Fold Change in RFU')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title("Fold Change in RFU for each gRNA position at 8 hours")
        plt.savefig("./Figures/foldchange.png")

    # Define the model parameters.
    stride_parameter = 20
    label_dim = 1
    embedding_dim = 18
    intermediate_dim = 50
    batch_size_parameter=300 #4000 for howard's e. coli dataset (from Enoch's code)
    debug_splash = 0
    this_step_size_val = 0.25
    this_corpus,this_labels = make_labeled_corpus(allseqs, data, stride_parameter)

    # Define the random transformation householder matrix.
    Rand_Transform = rvs(dim=stride_parameter)
    
    # Define the corpus for the model.
    this_corpus_vec = []
    for this_corpus_elem in this_corpus:
        vec_value = sequence_encoding(this_corpus_elem)
        vec_value = np.dot(Rand_Transform,vec_value)

        this_corpus_vec.append(vec_value)

    this_corpus_vec = np.asarray(this_corpus_vec)
    this_labels = np.expand_dims(this_labels,axis=1)
    hidden_vars_list = [embedding_dim, stride_parameter]


    # Define the tensorflow session
    model = create_model(input_shape=(stride_parameter,), embedding_dim=embedding_dim,
                         intermediate_dim=intermediate_dim, label_dim=label_dim)
