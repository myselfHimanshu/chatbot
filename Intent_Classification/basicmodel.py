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




        


    
    
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    