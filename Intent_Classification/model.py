import os, sys, json
import numpy as np
from scipy import io
from helpers.slotdata import dataSet, readNum
from helpers.embeddings import PredefinedEmbedding
from helpers.encoding import encoding
import argparse

from keras.preprocessing import sequence
from keras.models import Sequential, Graph, Model
from keras.layers import Input, merge, Merge, Dense, TimeDistributedDense, Dropout, Activation, ReapeatVector, Permute, Reshape, RepearVector, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
from keras.constraints import maxnorm, nonneg
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from History import LossHistory

#HYPERPARAMETERS
hidden_size =
training_file

class KerasModel(object):
    def __init__(self, argparams):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
