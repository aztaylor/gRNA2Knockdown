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

from sklearn.decomposition import PCA    
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

'''This module contains the functions to encode DNA sequences and for use in predicting  the knockdown 
efficiency of CRISPR dCasRx-gRNAs effectors. The model takes in the RNA sequence of the targeted transcript 
embedds the sequence using an autoencoder. Subsequently, the model takes the embedded sequence and the mRNA of the 
targetto predict the knockdown efficiency. The base design is adaptable with a variable number of hidden layers and 
units andis meant to be able to construct autoencoders, feedforward NNs and residual networks.

The model is trained using  the Adam optimizer and a custom VAE embbed loss function to determine the mean squared 
error.The original code was developed by Enoch Yeung in the Biological Control Laboratory at the University of 
California, Santa Barbara. Some things to note:
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
now = datetime.now()
date = now.strftime('%Y%m%d')
time = now.strftime('%H%M%S')


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
    scrabble the 4-hot encoding to project into random input space. This improves performance for reason I do not yet
    know.
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
    '''Calculate the embedding loss. The embedding loss accounts for the covarariance between the embeddings.
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

#HybridLoss = customRegressLoss(this_y_out,this_u,this_embedding,this_regress_y,this_regress_y_labels)
def customRegressLoss(y_model:tf.Variable, y_true:tf.Variable,
               embed_true:tf.Variable,regress_y_model:tf.Variable,regress_y_true:tf.Variable) -> tf.Variable:
    '''Custom loss function that combines the VAE loss and the embedding loss. The VAE loss is the mean squared error
    between the predicted and true y values. The embedding loss is the mean squared error between the predicted and true
    embeddings. .... needs more 
    args:
        y_model: tf.Variable, predicted y values
        y_true: tf.Variable, true y values
        embed_true: tf.Variable, true embeddings
        
    returns:
        tf.Variable, custom loss
        '''
    regression_loss = tf.norm(this_regress_y-this_regress_y_labels,axis=[0,1],ord=2)/tf.norm(this_regress_y_labels,axis=[0,1],ord=2)
    lambda_regression = 0.01
    return vae_loss(y_model,y_true)+embed_loss(y_true,embed_true) + lambda_regression*regression_loss



# Define the network
def network_assemble(input_var:tf.Variable, W_list:list, b_list:list, 
                     keep_prob=1.0, activation_flag=1, 
                     res_net=0, debug_splash=False)->(tf.Variable, list): # type: ignore
    ''''Assemble the network with the given weights and biases. The activation function is defined by the 
    activation_flag. The res_net flag is used to define if the network is a residual network or not.
    args:
        sess: tf.Session, tensorflow session
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
                prev_layer_output += tf.matmul(z_temp_list[-1], W1)+b1

            if activation_flag==1:
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
    return y_out, z_temp_list

#Train and test the network
def train_net(sess, u_all_training:np.array, u_feed:tf.Variable, y_all_training:np.array,y_feed:tf.Variable, 
              obj_func:tf.Variable, optimizer:tf.compat.v1.train.Optimizer,
              this_vae_loss:tf.Variable, this_embed_loss:tf.Variable, 
              valid_error_thres=1e-2, test_error_thres=1e-2, max_iters=1e6, 
              step_size_val=0.01, batchsize=10, samplerate=5000, good_start=1, 
              test_error=100.0, save_fig=None) -> list:
    '''Train the network using the Adam optimizer. The training is done in batches and the error is calculated for the
    training, validation and test sets. The training stops when the validation and test errors are below the threshold.
    args:
        sess: tf.Session, tensorflow session
        u_all_training: np.array, training data
        u_feed: tf.Variable, input Variable
        y_all_training: np.array, training data from plate reader 
        y_feed: tf.Variable, the placeholder variable that will store the training data from the plate reader (slices of y_all_training)
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

    while (((test_error>test_error_thres) or (valid_error > valid_error_thres)) 
            and iter < max_iters):
        iter+=1

        all_ind = set(np.arange(0,len(u_all_training)))
        select_ind = np.random.randint(0,len(u_all_training),size=batchsize)
        valid_ind = list(all_ind -set(select_ind))[0:batchsize]
        select_ind_test = list(all_ind - set(valid_ind) - 
                            set(select_ind))[0:batchsize]

        u_batch =[]
        u_valid = []
        u_test_train = []

        y_batch = []
        y_valid = []
        y_test_train = []
        
        for j in range(0,len(select_ind)):
            u_batch.append(u_all_training[select_ind[j]])
            y_batch.append(y_all_training[select_ind[j]])

        for k in range(0,len(valid_ind)):
            u_valid.append(u_all_training[valid_ind[k]])
            y_valid.append(y_all_training[valid_ind[k]])

        for k in range(0,len(select_ind_test)):
            u_test_train.append(u_all_training[select_ind_test[k]])
            y_test_train.append(y_all_training[select_ind_test[k]])

        optimizer.run(feed_dict={u_feed:u_batch,y_feed:y_batch}, session=sess) # embed_feed:,step_size:step_size_val});
        valid_error = obj_func.eval(feed_dict={u_feed:u_valid,y_feed:y_valid}, session=sess) # embed_feed:y_valid});

        test_error = obj_func.eval(feed_dict={u_feed:u_test_train,y_feed:y_test_train}, 
                                   session=sess) # embed_feed:y_test_train});


        if iter%samplerate==0:
            training_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_batch, y_feed:y_batch}, session=sess));
            validation_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_valid, y_feed:y_valid}, session=sess));
            test_error_history_nocovar.append(obj_func.eval(
                feed_dict={u_feed:u_test_train, y_feed:y_test_train}, session=sess));


        if (iter%1000==0) or (iter==1):
            print("\r step %d , validation error %g"%(iter, obj_func.eval(
                    feed_dict={u_feed:u_valid, y_feed:y_batch}, session=sess)) );
            
            print("\r step %d , test error %g"%(iter, obj_func.eval(
                    feed_dict={u_feed:u_test_train, y_feed:y_test_train}, session=sess)) );
            print("\r Reconstruction Loss: " + repr(this_vae_loss.eval(
                    feed_dict={u_feed:u_all_training, y_feed:y_all_training}, session=sess)) );
            print("\r Embedding Loss: " + repr(this_embed_loss.eval(
                    feed_dict={u_feed:u_all_training, y_feed:y_all_training}, session=sess)) );
    all_histories = [training_error_history_nocovar, 
                    validation_error_history_nocovar,
                    test_error_history_nocovar,
                    u_all_training]

    if save_fig is not None:
        fig, ax = plt.subplots(1,1)
        x = np.arange(0,len(validation_error_history_nocovar),1)
        ax.plot(x,training_error_history_nocovar,label='train. err.')
        ax.plot(x,validation_error_history_nocovar,label='valid. err.')
        ax.plot(x,test_error_history_nocovar,label='test err.')
        #ax.plot(x,u_all_training,label='Reconstruction Loss')
        ax.legend()
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Error')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(f'ErrorHistory{date}_{time}')
        
        plt.savefig(save_fig)
        plt.close()
    return all_histories, good_start

SeqMap = ['A','C','G','T']

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
    #opt_index = np.argmin(np.array(seq_dist_list))
    return(SeqMap[opt_index])

def vecback2seq(untransformed_vec):
    '''Convert the lattice-like vector representation/encoding to a DNA sequence. The encoding is mapped as to be
        normalized between 0 and 1.
        args:
            untransformed_vec: np.array, lattice-like vector representation/encoding
        returns:
            list, list of DNA sequences
    '''
    #untransformed_vec is computed using the inverse of Rand_Transform to recover lattice-like vector representation/
    #encoding
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

    NumRowsonPlate = 8
    NumColumnsonPlate = 12
    HourHorizon = 18
    SamplingRate = 3/60; 
    data0, time0 = pr.Organize(data_0nM_fp,NumRowsonPlate,NumColumnsonPlate,HourHorizon,SamplingRate)
    data1, time1 = pr.Organize(data_10mM_fp,NumRowsonPlate,NumColumnsonPlate,HourHorizon,SamplingRate)

    this_fig = plt.figure()
    OD_key = list(data0.keys())[0]
    FL_key = list(data0.keys())[1]

    this_baseline_od_data = data0[OD_key]
    this_baseline_fl_data = data0[FL_key]
    this_induced_od_data = data1[OD_key]
    this_induced_fl_data = data1[FL_key]
    
    this_time = time0[FL_key]
    foldchangedata = data0[FL_key]-data0[FL_key] # makes a zeros matrix with the right dimensions for storing fold change data 
    for row in range(0,8):
        for col in range(0,12):
            print(this_induced_fl_data.shape)
            print(this_time.shape)
            odnormfl_induced = this_induced_fl_data[row][col]/this_induced_od_data[row][col]
            odnormfl_baseline = this_baseline_fl_data[row][col]/this_baseline_od_data[row][col]
            this_foldchange = odnormfl_induced/odnormfl_baseline
            foldchangedata[row,col,:] = this_foldchange

            plt.scatter(this_time,this_foldchange)
            
    this_fig.savefig(f'QualityDatafromAlec{date}_{time}.eps')

    listed_foldchangedata = foldchangedata.reshape(np.int(foldchangedata.shape[0]*foldchangedata.shape[1]),foldchangedata.shape[2])
    
    print("listed fold change data shape: " + repr(listed_foldchangedata.shape))
    


    
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
        plt.savefig(f"./Figures/foldchange{date}_{time}.png")

    # Define the model parameters.
    stride_parameter = 30
    label_dim = 1
    embedding_dim = 15
    outpuDim = int(HourHorizon*1/SamplingRate)
    feedforwardDepth = 2
    feedforwardDim = 100
    intermediate_dim = 100
    batch_size_parameter=20 #4000 for howard's e. coli dataset (from Enoch's code)
    debug_splash = 0
    this_step_size_val = 0.01
    max_iters = 1e3
    this_corpus,this_labels = make_labeled_corpus(allseqs, data,
                                                  stride_parameter)

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
    n_pre_post_layers = 10; 
    hidden_vars_list = [intermediate_dim]*n_pre_post_layers+[embedding_dim]+\
        [intermediate_dim]*n_pre_post_layers+[stride_parameter]
    if False:
        print(hidden_vars_list)

    # Define the tensorflow session
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution() # needed because of placeholder variables



    # Define the placeholders for the input sequences
    this_u = tf.compat.v1.placeholder(tf.float32, 
                                      shape=[None,stride_parameter])
    # Define placeholder for regression label (vectors made from time-series traces of plate reader data)
    this_regress_y_labels = tf.compat.v1.placeholder(tf.float32,shape=[None,outpuDim])

    # Instantiate the autoencoder network 
    with tf.device('/gpu:0'):
        this_W_list,this_b_list = initialize_Wblist(stride_parameter,
                                                    hidden_vars_list)
        this_y_out,all_layers = network_assemble(this_u,this_W_list,this_b_list
                                                ,keep_prob=1.0,
                                                activation_flag=2,res_net=0)

    # Define a handle that accesses the embedding layer 
    this_embedding = all_layers[n_pre_post_layers+1]
    # Define the regression network depth, width, and output dimension:     
    regress_list = [feedforwardDim]*feedforwardDepth+[outpuDim]
    # I believe this is the regression part of the network
    with tf.device('/gpu:0'):
        this_Wregress_list,this_bregress_list = initialize_Wblist(embedding_dim,
                                                                  regress_list)
        this_regress_y,all_regress_layers = network_assemble(this_embedding,this_Wregress_list,this_bregress_list)
        
        HybridLoss = customRegressLoss(this_y_out,this_u,this_embedding,this_regress_y,this_regress_y_labels)

        result = sess.run(tf.compat.v1.global_variables_initializer())
        this_optim = tf.compat.v1.train.AdagradOptimizer(
            learning_rate=this_step_size_val).minimize(HybridLoss)
        step_size = tf.compat.v1.placeholder(tf.float32,shape=[])
        result = sess.run(tf.compat.v1.global_variables_initializer())
        this_vae_loss = vae_loss(this_y_out,this_u)
        this_embed_loss = embed_loss(this_u,this_embedding)

        if True:
            # Train the network
            train_figure_name = f"Figures/Training{embedding_dim}_\
                {intermediate_dim}_{stride_parameter}_{n_pre_post_layers}.png"
                
            train_net(sess, this_corpus_vec,this_u,listed_foldchangedata,this_regress_y_labels,HybridLoss,
                    this_optim,
                    this_vae_loss=this_vae_loss,
                    this_embed_loss=this_embed_loss,  
                    batchsize=batch_size_parameter,
                    step_size_val=this_step_size_val,
                    max_iters=max_iters,
                    save_fig= train_figure_name)
    # This is likely redudent code that can be removed.
    # I think we just need to reconsider the loss function at this point.
    if False:
        feedforwardList = [embedding_dim]+[feedforwardDim]*feedforwardDepth+\
            [outpuDim]
        with tf.device('/gpu:0'):
            Wfeedforward, bfeedforward = initialize_Wblist(embedding_dim,
                                                            feedforwardList)
            y_out,all_layers = network_assemble(this_embedding,Wfeedforward,bfeedforward)
            this_vae_loss = vae_loss(y_out,this_embedding)
        
    all_mismatches = []
    for ind in range(0,len(this_corpus_vec)):
        z_ind = this_y_out.eval(feed_dict={this_u:[this_corpus_vec[ind]]},\
            session=sess)
        this_seq_out = vecback2seq(np.dot(np.linalg.inv(Rand_Transform),z_ind.T))
        print("Predicted:"+repr("".join(this_seq_out))[0:11])
        print("Ground Truth:"+repr("".join(this_corpus[ind][0:10])))
        print("\n")
        this_seq_out = ''.join(this_seq_out)
        all_mismatches.append(num_mismatch(this_seq_out,this_corpus[ind]))
    
    mismatch_process = np.array(all_mismatches)
    np.sum(mismatch_process)/(len(mismatch_process)*1.0)
    fig, ax = plt.subplots(1,1)
    ax.hist(mismatch_process, bins=range(0,31,1))
    ax.set_xlabel("Number of Mismatches")
    ax.set_ylabel("Frequency")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title("Number of Mismatches in Predicted Sequences")
    plt.savefig(f"Figures/mismatches{date}_{time}.png")

    subset_embeddings = this_embedding.eval(feed_dict={this_u:this_corpus_vec},
                                            session=sess)

    X = subset_embeddings
    pca = PCA(n_components=3)
    pca.fit(X)
    PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)

    print("PCA Explained Variance Ratio: " + repr(pca.explained_variance_ratio_))
    print("PCA Singular Values: " + repr(pca.singular_values_))

    
    X_transformed = pca.transform(X)
    X_transformed = X_transformed[0:]

    X_transformed.shape

    # Fixing random state for reproducibility
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    this_colors = 0*np.random.rand(len(X_transformed),3)
    print(this_labels)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for x_ind in range(0,len(X_transformed)):
        x= X_transformed[x_ind][0]
        y= X_transformed[x_ind][1]
        z= X_transformed[x_ind][2]
        if this_labels[x_ind]>0.66:
            this_colors[x_ind][0] = this_labels[x_ind]/np.max(this_labels)
        if 0.66>this_labels[x_ind]>0.33:
            this_colors[x_ind][1] = this_labels[x_ind]/np.max(this_labels)
        if 0.33>this_labels[x_ind]>-10.0:
            this_colors[x_ind][2] = this_labels[x_ind]/np.max(this_labels)


    ax.scatter(X_transformed[:,0], X_transformed[:,1],X_transformed[:,2],
               c=this_colors, marker='o',alpha=0.25)
    ax.view_init(30, azim=240)
    ax.set_xlabel('Principal Component One')
    ax.set_ylabel('Principal Component Two')
    ax.set_zlabel('Principal Component Three')
    plt.tight_layout()
    fig.savefig(f"Figures/PCA{date}_{time}.png")
