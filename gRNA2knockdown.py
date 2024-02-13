import csv
import math
import random
import numpy as np

import tensorflow as tf

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

def weight(shape: tuple, name: str, initializer: tf.initializer) -> tf.Variable:
    '''Create a weight variable with a given shape and name.
    args:
        shape: tuple, shape of the weight
        name: str, name of the weight
        initializer: tf.initializer, initializer of the weight
    returns:
        tf.Variable, tensorflow variable
    '''
    return tf.get_variable(name, shape, initializer=initializer)