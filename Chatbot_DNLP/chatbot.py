#Building a chatbot with Deep NLP

#Importing the libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import re
import time


#####PART I - Data Preprocessing#####

#Importing the dataset
data_path = "DATA/movie_dialogues/"

lines = open(data_path+'movie_lines.txt',encoding='utf-8', errors='ignore').read().split("\n")
conversations = open(data_path+'movie_conversations.txt',encoding='utf-8', errors='ignore').read().split("\n")

#Creating the dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(" +++$+++ ")
    if len(_line)==5:
        id2line[_line[0]] = _line[-1]

#Creating the list of the conversations
conversations_ids = []
for conv in conversations[:-1]:
    _conv = conv.split(" +++$+++ ")[-1][1:-1].replace("'","").replace(" ","")
    conversations_ids.append(_conv.split(","))

#Getting seperately the questions and answers




























