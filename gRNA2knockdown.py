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

'''This module contains the functions to encode DNA sequences and to use those encodeding to predict the knockdown efficiency of CRISPR 
CasRx gRNAs. The model takes in the RNA sequence of the targeted transcript and the gRNA sequence and outputs the knockdown efficiency
in terms of '''

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

def create_corpus(seq_array:np.ndarray, y_trace: np.ndarray, stride = 1) -> dict:
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


def xavier_init(n_inputs: int, n_outputs: int, uniform=True) -> tf.initializer:
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
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

def weight_variable(shape: tuple) -> tf.Variable:
    '''Create a weight variable with a given shape and name. Defined to be used in the standard definitiion
    of a neuron: Wx + b, where W is the weight, x is the input and b is the bias.
    args:
        shape: tuple, shape of the weight
    returns:
        tf.Variable, tensorflow variable
    '''
    std_dev = math.sqrt(3.0 /(shape[0] + shape[1]))
    return tf.Variable(tf.random.truncated_normal(shape, mean=0.0,stddev=std_dev,dtype=tf.float32))

def bias_variable(shape) -> tf.Variable:
    '''Create a bias variable with a given shape and name. Defined to be used in the standard definitiion
    of a neuron: Wx + b, where W is the weight, x is the input and b is the bias.
    args:
        shape: tuple, shape of the bias
    returns:
        tf.Variable, tensorflow variable'''
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.random.truncated_normal(shape, mean=0.0,stddev=std_dev,dtype=tf.float32))

def initialize_Wblist(n_u,hv_list):
    W_list = [];
    b_list = [];
    n_depth = len(hv_list);
    print("Length of hv_list: " + repr(n_depth))
    #hv_list[n_depth-1] = n_y;
    for k in range(0,n_depth):

        if k==0:
            W1 = weight_variable([n_u,hv_list[k]]);
            b1 = bias_variable([hv_list[k]]);
            W_list.append(W1);
            b_list.append(b1);
        else:
            W_list.append(weight_variable([hv_list[k-1],hv_list[k]]));
            b_list.append(bias_variable([hv_list[k]]));
    result = sess.run(tf.compat.v1.global_variables_initializer())
    return W_list,b_list;

def network_assemble(input_var:tf.Variable, W_list:list, b_list:list, keep_prob=1.0, activation_flag=1, res_net=0)->(tf.Variable, list):
    ''''Assemble the network with the given weights and biases. The activation function is defined by the activation_flag. The res_net
    flag is used to define if the network is a residual network or not.
    args:
        input_var: tf.Variable, input variable
        W_list: list, list of weights
        b_list: list, list of biases
        keep_prob: float, dropout rate
        activation_flag: int, flag to define the activation function
        res_net: int, flag to define if the network is a residual network
    returns:
        y_out: tf.Variable, output of the network
        z_temp_list: list, list of activations
    '''
    n_depth = len(W_list)
    print("n_depth: " + repr(n_depth))
    z_temp_list = []

    for k in range(0,n_depth):
        # form the input layer with the flag variable determining the activation function.
        if (k==0):
            W1 = W_list[0]
            b1 = b_list[0]
            if activation_flag==1:# RELU
                z1 = tf.nn.dropout(tf.nn.relu(tf.matmul(input_var,W1)+b1),rate=1 - (keep_prob))
            if activation_flag==2: # ELU
                z1 = tf.nn.dropout(tf.nn.elu(tf.matmul(input_var,W1)+b1),rate=1 - (keep_prob))
            if activation_flag==3: # tanh
                z1 = tf.nn.dropout(tf.nn.tanh(tf.matmul(input_var,W1)+b1),rate=1 - (keep_prob))
            z_temp_list.append(z1)
        # form the hidden layers with the flag variable determining the activation function.
        if not (k==0) and k < (n_depth-1):
            prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]
            if res_net and k==(n_depth-2):
                prev_layer_output += tf.matmul(u,W1)+b1 #  this expression is not compatible for variable width nets (where each layer has a different width at inialization - okay with regularization and dropout afterwards though)
            if activation_flag==1:
                z_temp_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output),rate=1 - (keep_prob)))
            if activation_flag==2:
                z_temp_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output),rate=1 - (keep_prob)))
            if activation_flag==3:
                z_temp_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output),rate=1 - (keep_prob)))
        # form the output layer with the flag variable determining the activation function.
        if not (k==0) and k == (n_depth-1):
            prev_layer_output = tf.matmul(z_temp_list[k-1],W_list[k])+b_list[k]
            z_temp_list.append(prev_layer_output)

    if debug_splash:
        print("[DEBUG] z_list" + repr(z_list[-1]))

    y_out = z_temp_list[-1]

    result = sess.run(tf.compat.v1.global_variables_initializer())
    return y_out, z_temp_list

