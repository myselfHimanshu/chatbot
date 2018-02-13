import os, sys, json
import numpy as np
from scipy import io
import re
import argparse

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, merge, Merge, Dense, Dropout, Activation, RepeatVector, Permute, Reshape, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam
from keras.constraints import maxnorm, nonneg
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

#DataReading
class DataSet(object):
    def __init__(self, datafile, toggle, wordDictionary, tagDictionary, id2word, id2tag):
        if toggle=="train":
            self.dataset = readData(datafile)
        if toggle=="val":
            self.dataset = readTest(datafile, wordDictionary, tagDictionary, id2word, id2tag)
        if toggle=="test":
            self.dataset = readTest(datafile, wordDictionary, tagDictionary, id2word, id2tag)
            
def readData(dataFile):
    utterances = list()
    tags = list()
    starts = list()
    startid = list()
    
    word_vocab_index = 2
    tag_vocab_index = 2
    word2id = {'<pad>':0,'<unk>':1}
    tag2id = {'<pad>':0,'<unk>':1}
    id2word = ['<pad>','<unk>']
    id2tag = ['<pad>','<unk>']
    
    utt_count = 0
    temp_startid = 0
    for line in open(dataFile,'r'):
        d = line.split('\t')
        utt = d[0].strip()
        t = d[1].strip()
        if len(d)>2:
            start = np.bool(int(d[2].strip()))
            starts.append(start)
            if start:
                temp_startid = utt_count
            startid.append(temp_startid)
            
        temp_utt = list()
        temp_tags = list()
        mywords = utt.split()
        mytags = t.split()
        if len(mywords)!=len(mytags):
            print(mywords)
            print(mytags)
        
        for i in range(len(mywords)):
            if mywords[i] not in word2id:
                word2id[mywords[i]]=word_vocab_index
                id2word.append(mywords[i])
                word_vocab_index+=1
            if mytags[i] not in tag2id:
                tag2id[mytags[i]]=tag_vocab_index
                id2tag.append(mytags[i])
                tag_vocab_index+=1
                
            temp_utt.append(word2id[mywords[i]])
            temp_tags.append(tag2id[mytags[i]])
        utt_count+=1
        utterances.append(temp_utt)
        tags.append(temp_tags)
        
    data = {'start':starts,'startid':startid,'utterances':utterances, 'tags':tags, 'uttCount':utt_count, 'id2word':id2word, 'id2tag':id2tag, 'wordVocabSize':word_vocab_index, 'tagVocabSize':tag_vocab_index, 'word2id':word2id, 'tag2id':tag2id}
    return data


def readTest(testFile, word2id, tag2id, id2word, id2tag):
    utterances = list()
    tags = list()
    starts = list()
    startid = list()
    
    utt_count = 0
    temp_startid = 0
    for line in open(testFile,'r'):
        d = line.split('\t')
        utt = d[0].strip()
        t = d[1].strip()
        if len(d)>2:
            start = np.bool(int(d[2].strip()))
            starts.append(start)
            if start:
                temp_startid = utt_count
            startid.append(temp_startid)
            
        temp_utt = list()
        temp_tags = list()
        mywords = utt.split()
        mytags = t.split()
        if len(mywords)!=len(mytags):
            print(mywords)
            print(mytags)
        
        for i in range(len(mywords)):
            if mywords[i] not in word2id:
                temp_utt.append(1)
            else:
                temp_utt.append(word2id[mywords[i]])
            if mytags[i] not in tag2id:
                temp_tags.append(1)
            else:
                temp_tags.append(tag2id[mytags[i]])
        utt_count+=1
        utterances.append(temp_utt)
        tags.append(temp_tags)
        wordVocabSize = len(word2id)
        
    data = {'start': starts, 'startid': startid, 'utterances': utterances, 'tags': tags, 'uttCount': utt_count, 'wordVocabSize' : wordVocabSize, 'id2word':id2word, 'id2tag': id2tag}
    return data

def readNum(numFile):
    numlist = map(int, file(numfile).read().strip().split())
    totalList = list()
    cur = 0
    for num in numlist:
        cur+=num+1
        totalList.append(cur)
    return numlist, totalList


def encoding(data, encode_type, time_length, vocab_size):
    if encode_type=="1hot":
        return onehot_encoding(data, time_length, vocab_size)
    elif encode_type=="embedding":
        return data
    
def onehot_encoding(data, time_length, vocab_size):
    X = np.zeros((len(data), time_length, vocab_size), dtype=np.bool)
    for i,sent in enumerate(data):
        for j,k in enumerate(sent):
            X[i,j,k]=1
    return X

def onehot_sent_encoding(data, vocab_size):
    X = np.zeros((len(data), vocab_size), dtype=np.bool)
    for i, sent in enumerate(data):
        for j,k in enumerate(sent):
            X[i,k] = 1
    return X

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        

class KerasModel(object):
    def __init__(self, argparams):
        self.hidden_size = argparams['hidden_size'] # size of hidden layer of neurons 
        self.learning_rate = argparams['learning_rate']
        self.training_file = argparams['train_data_path']
        self.validation_file = argparams['dev_data_path']
        self.test_file = argparams['test_data_path']
        self.result_path = argparams['result_path']
        self.train_numfile = argparams['train_numfile']
        self.dev_numfile = argparams['dev_numfile']
        self.test_numfile = argparams['test_numfile']
        self.update_f = argparams['sgdtype'] # options: adagrad, rmsprop, vanilla. default: vanilla
        self.decay_rate = argparams['decay_rate'] # for rmsprop
        self.default = argparams['default_flag'] # True: use defult values for optimizer
        self.momentum = argparams['momentum'] # for vanilla update
        self.max_epochs = argparams['max_epochs']
        self.activation = argparams['activation_func'] # options: tanh, sigmoid, relu. default: relu
        self.smooth_eps = argparams['smooth_eps'] # for adagrad and rmsprop
        self.batch_size = argparams['batch_size']
        self.input_type = argparams['input_type'] # options: 1hot, embedding, predefined
        self.emb_dict = argparams['embedding_file']
        self.embedding_size = argparams['embedding_size']
        self.dropout = argparams['dropout']
        self.dropout_ratio = argparams['dropout_ratio']
        self.iter_per_epoch = argparams['iter_per_epoch']
        self.arch = argparams['arch']
        self.init_type = argparams['init_type']
        self.fancy_forget_bias_init = argparams['forget_bias']
        self.time_length = argparams['time_length']
        self.his_length = argparams['his_length']
        self.mdl_path = argparams['mdl_path']
        self.log = argparams['log']
        self.record_epoch = argparams['record_epoch']
        self.load_weight = argparams['load_weight']
        self.combine_his = argparams['combine_his']
        self.time_decay = argparams['time_decay']
        self.shuffle = argparams['shuffle']
        self.set_batch = argparams['set_batch']
        self.tag_format = argparams['tag_format']
        self.e2e_flag = argparams['e2e_flag']
        self.output_att = argparams['output_att']
        self.sembedding_size = self.embedding_size
        self.model_arch = self.arch
        
        if self.validation_file is None:
            self.nodev = True
        else:
            self.nodev = False
        if self.input_type=="embedding":
            self.model_arch = self.model_arch + "emb"
        if self.time_decay:
            self.model_arch = self.model_arch + "+T"
        if self.e2e_flag:
            self.model_arch = 'e2e-' + self.model_arch
            
    def test(self, H, X, data_type, tadDict, pad_data):
        if self.default:
            target_file = self.result_path + '/' + self.model_arch + '_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type
        else:
            target_file = self.result_path + '/' + self.model_arch +'-LR-'+str(self.learning_rate)+'_H-'+str(self.hidden_size)+'_O-'+self.update_f+'_A-'+self.activation+'_WR-'+self.input_type
        
        if 'memn2n' in self.arch or self.arch[0] == 'h':
            batch_data = [H, X]
        else:
            batch_data = X
            
        






































        


    
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    