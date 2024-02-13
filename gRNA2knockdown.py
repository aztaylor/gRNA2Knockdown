#!/usr/bin/env python3.11
__author__ = "Alec Taylor, Enoch Yeung"
__version__ = "1.0.0"
__maintainer__ = "Alec Taylor"
__email__ = "aztaylor76@fastmail.com"
__status__ = "Development"

import csv
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''This module contains the functions to encode DNA sequences and to use those encodeding to predict the knockdown 
efficiency of CRISPR CasRx gRNAs. The model takes in the RNA sequence of the targeted transcript and the gRNA sequence
and outputs the knockdown efficiencyin terms of marker fold change. The model is a feedforward neural network with a
Variable number of hidden layers and units. Optionally, it can be a residual network. The model is trained using the 
Adam optimizer and the loss function is the mean squared error. The original code was developed by Enoch Yeung in the 
Biological Control Laboratory at the University of California, Santa Barbara.
'''
# Start a tensorflow session
sess = tf.compat.v1.Session()

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

def create_corpus(seq_array:np.ndarray, y_trace: np.ndarray, stride=1) -> dict:
    '''Create a corpus of sequences and their corresponding labels (y_trace).
    args:
        seq_array: np.ndarray, 2D array of sequences
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
    return corpus


def xavier_init(n_inputs: int, n_outputs: int, uniform=True) -> tf.initializers:
    '''Initialize weights with Xavier initialization. From Enoch's code this initialize
    the weights with a uniform distribution to keep the scale if gradients roughly the same in all layers.
    From originally Xavier Glorot and Yoshua Bengio (2010).
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

def bias_Variable(shape) -> tf.Variable:
    '''Create a bias Variable with a given shape and name. Defined to be used in the standard definitiion
    of a neuron: Wx + b, where W is the weight, x is the input and b is the bias.
    args:
        shape: tuple, shape of the bias
    returns:
        tf.Variable, tensorflow Variable
    '''
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.random.truncated_normal(shape, mean=0.0,
                                                  stddev=std_dev,
                                                  dtype=tf.float32))

def initialize_Wblist(n_u,hv_list) -> (list, list): # type: ignore
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
    result = sess.run(tf.compat.v1.global_Variables_initializer())
    return W_list,b_list

def rvs(dim=3):
    '''Generate a random orthogonal matrix. The matrix is generated using the Householder transformation.
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
        tf.Variable, custom loss'''
    return vae_loss(y_model,y_true)+embed_loss(y_true,embed_true)


# Define the network
def network_assemble(input_var:tf.Variable, W_list:list, b_list:list, 
                     keep_prob=1.0, activation_flag=1, 
                     res_net=0, debug_splash=False)->(tf.Variable, list): # type: ignore
    ''''Assemble the network with the given weights and biases. The activation function is defined by the 
    activation_flag. The res_net flag is used to define if the network is a residual network or not.
    args:
        input_var: tf.Variable, input Variable
        W_list: list, list of weights
        b_list: list, list of biases
        keep_prob: float, dropout rate
        activation_flag: int, flag to define the activation function
        res_net: int, flag to define if the network is a residual network
        debug_splash: bool, flag to print debug information
    returns:
        y_out: tf.Variable, output of the network
        z_temp_list: list, list of activations
    '''
    n_depth = len(W_list)
    print("n_depth: " + repr(n_depth))
    z_temp_list = []

    for k in range(0,n_depth):
        # form the input layer with the flag Variable determining the activation function.
        if (k==0):
            W1 = W_list[0]
            b1 = b_list[0]
            if activation_flag==1:# RELU
                z1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input_var,W1)+b1)
                                   ,rate=1 - (keep_prob))
            if activation_flag==2: # ELU
                z1 = tf.nn.dropout(tf.nn.elu(tf.matmul(input_var,W1)+b1),
                                   rate=1 - (keep_prob))
            if activation_flag==3: # tanh
                z1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(input_var,W1)+b1),
                                   rate=1 - (keep_prob))
            z_temp_list.append(z1)
        # form the hidden layers with the flag Variable determining the activation function.
        if not (k==0) and k < (n_depth-1):
            prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]
            if res_net and k==(n_depth-2):
            # this expression is not compatible for Variable width nets (where each layer has a different width at 
            # inialization - okay with regularization and dropout afterwards though)
                prev_layer_output += tf.matmul(u,W1)+b1
            if activation_flag==1:
                prev_layer_output += tf.matmul(u,W1)+b1 
                z_temp_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output),
                                                 rate=1 - (keep_prob)))
            if activation_flag==2:
                z_temp_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output),
                                                 rate=1 - (keep_prob)))
            if activation_flag==3:
                z_temp_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output),
                                                 rate=1 - (keep_prob)))
        # form the output layer with the flag Variable determining the activation function.
        if not (k==0) and k == (n_depth-1):
            prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]
            z_temp_list.append(prev_layer_output)

    if debug_splash:
        print("[DEBUG] z_list" + repr(z_temp_list[-1]))

    y_out = z_temp_list[-1]

    result = sess.run(tf.compat.v1.global_Variables_initializer())
    return y_out, z_temp_list

#Train and test the network
def train_net(u_all_training:np.array, u_feed:tf.Variable, obj_func:tf.Variable,
              optimizer:tf.Variable, u_control_all_training=None, 
              valid_error_thres=1e-2, test_error_thres=1e-2,
              max_iters=100000, step_size_val=0.01, batchsize=10, 
              samplerate=5000, good_start=1, val_error=100.0, 
              test_error=100.0) -> list:
    '''Train the network using the Adam optimizer. The training is done in batches and the error is calculated for the
    training, validation and test sets. The training stops when the validation and test errors are below the threshold.
    args:
        u_all_training: np.array, training data
        u_feed: tf.Variable, input Variable
        obj_func: tf.Variable, objective function
        optimizer: tf.Variable, optimizer
        u_control_all_training: np.array, control training data
        valid_error_thres: float, validation error threshold
        test_error_thres: float, test error threshold
        max_iters: int, maximum number of iterations
        step_size_val: float, step size value
        batchsize: int, batch size
        samplerate: int, sample rate
        good_start: int, flag to determine if the training has a good start
        val_error: float, validation error
        test_error: float, test error
    returns:
        all_histories: list, list of error histories
        good_start: int, flag to determine if the training has a good start
    '''

    iter = 0
    training_error_history_nocovar = []
    validation_error_history_nocovar = []
    test_error_history_nocovar = []

    training_error_history_withcovar = []
    validation_error_history_withcovar = []
    test_error_history_withcovar = []


    while (((test_error>test_error_thres) or (valid_error > valid_error_thres)) 
            and iter < max_iters):
        iter+=1

        all_ind = set(np.arange(0,len(u_all_training)))
        select_ind = np.random.randint(0,len(u_all_training),size=batchsize)
        valid_ind = list(all_ind -set(select_ind))[0:batchsize]
        select_ind_test = list(all_ind - set(valid_ind) - 
                            set(select_ind))[0:batchsize]


        u_batch =[]
        u_control_batch = []

        u_valid = []
        u_control_valid = []

        u_test_train = []
        u_control_train = []

        u_control_test_train = []

        for j in range(0,len(select_ind)):
            u_batch.append(u_all_training[select_ind[j]])


        for k in range(0,len(valid_ind)):
            u_valid.append(u_all_training[valid_ind[k]])

        for k in range(0,len(select_ind_test)):
            u_test_train.append(u_all_training[select_ind_test[k]])

        optimizer.run(feed_dict={u_feed:u_batch}) # embed_feed:,step_size:step_size_val});
        valid_error = obj_func.eval(feed_dict={u_feed:u_valid}) # embed_feed:y_valid});
        test_error = obj_func.eval(feed_dict={u_feed:u_test_train}) # embed_feed:y_test_train});


        if iter%samplerate==0:
            training_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_batch}))#,embed_feed:y_batch}));
            validation_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_valid}))#,embed_feed:y_valid}));
            test_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_test_train}))#,embed_feed:y_test_train}));


        if (iter%10==0) or (iter==1):
            print ("step %d , validation error %g"%(iter, obj_func.eval(
                    feed_dict={u_feed:u_valid})))#,embed_feed:y_valid})));
            print ("step %d , test error %g"%(iter, obj_func.eval(
                    feed_dict={u_feed:u_test_train})));#,embed_feed:y_test_train})));
            print("Reconstruction Loss: " + repr(this_vae_loss.eval(
                    feed_dict={this_u:this_corpus_vec})))
            print("Embedding Loss: " + repr(this_embed_loss.eval(
                    feed_dict={this_u:this_corpus_vec})) )

    all_histories = [training_error_history_nocovar, 
                    validation_error_history_nocovar,test_error_history_nocovar]

    plt.close()
    x = np.arange(0,len(validation_error_history_nocovar),1)
    plt.plot(x,training_error_history_nocovar,label='train. err.')
    plt.plot(x,validation_error_history_nocovar,label='valid. err.')
    plt.plot(x,test_error_history_nocovar,label='test err.')
    plt.savefig('all_error_history.pdf')

    plt.close()
    return all_histories, good_start

# Run if not imported
if __name__ == "__main__":
    # Load the data
    with open('data.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data)
    seq_array = data[:,0]
    y_trace = data[:,1]

    # Create the corpus
    corpus = create_corpus(seq_array, y_trace)

    # Define the model
    n_x = 3330
    n_u = 5000
    n_y = 270


