import csv
import math
import random
import numpy as np

import tensorflow as tf

def sequence_encoding(sequence: str):
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

def create_corpus(seq_array:np.ndarray, y_trace: np.ndarray, stride = 1):
    '''Create a corpus of sequences and their corresponding labels (y_trace).
    args:
        seq_array: np.ndarray, 2D array of sequences
        y_trace: np.ndarray, 2D array of y traces (OD values or Fluorescence)
    returns:
        labeled_corpus: list, list of tuples (s0equence, y_trace)
    '''
    corpus = []
    for seq in seq_array:
        for i in range(0, len(seq)-23, stride):
            corpus.append(sequence_encoding(seq[i:i+23]))
    return corpus

