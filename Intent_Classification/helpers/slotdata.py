import re
import numpy as np

class dataSet(object):

    def __init__(self, dataFile, toggle, wordDictionary, tagDictionary, id2word, id2tag):
        if toggle=="train":
            self.dataSet = readData(dataFile)
        if toggle=="val":
            self.dataSet = readTest(dataFile, wordDictionary, tagDictionary, id2word, id2tag)
        if toggle=="test":
            self.dataSet = readTest(dataFile, wordDictionary, tagDictionary, id2word, id2tag)

    def getNum(self, numFile):
        return readNum(numFile)

    def getWordVocabSize(self):
        return self.dataSet["wordVocabSize"]

    def tagWordVocabSize(self):
        return self.dataSet["tagVocabSize"]

    def getNoExamples(self):
        return self.dataSet["uttCount"]

    def getExampleUtterance(self, index):
        return self.dataSet["utterances"][index]

    def getExampleTags(self,index):
        return self.dataSet['tags'][index]

    def getWordVocab(self):
        return self.dataSet['word2id']

    def getTagVocab(self):
        return self.dataSet['tag2id']

    def getIndex2Word(self):
        return self.dataSet['id2word']

    def getIndex2Tag(self):
        return self.dataSet['id2tag']

    def getTagAtIndex(self,index):
        return self.dataSet['id2tag'][index]

    def getWordAtIndex(self,index):
        return self.dataSet['id2word'][index]

    def getSample(self,batchSize):
        inputs={}
        targets={}
        indices = np.random.randint(0,self.dataSet['uttCount'],size=batchSize)
        for i in xrange(batchSize):
            inputs[i] = self.dataSet['utterances'][indices[i]]
            targets[i] = self.dataSet['tags'][indices[i]]

        return inputs,targets

def readData(dataFile):
    utterances = list()
    tags = list()
    starts = list()
    startid = list()

    word_vocab_index = 2
    tag_vocab_index = 2
    word2id = {"<pad>":0, "<unk>":1}
    tag2id = {"<pad>":0, "<unk>":1}
    id2word = ["<pad>","<unk>"]
    id2tag = ["<pad>","<unk>"]

    utt_count = 0
    temp_startid = 0
    for line in open(dataFile, 'r'):
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

        for i in xrange(len(mywords)):
            if mywords[i] not in word2id:
                word2id[mywords[i]] = word_vocab_index
                id2word.append(mywords[i])
                word_vocab_index += 1
            if mytags[i] not in tag2id:
                tag2id[mytags[i]] = tag_vocab_index
                id2tag.append(mytags[i])
                tag_vocab_index += 1

            temp_utt.append(word2id[mywords[i]])
            temp_tags.append(tag2id[mytags[i]])
        utt_count += 1
        utterances.append(temp_utt)
        tags.append(temp_tags)

    data = {'start':starts, 'startid':startid, 'utterances':utterances, 'tags':tags, 'uttCount':utt_count, 'id2word':id2word, 'id2tag':id2tag, 'wordVocabSize':word_vocab_index, 'tagVocabSize':tag_vocab_index, 'word2id':word2id, 'tag2id':tag2id }
    return data

def readTest(testFile, word2id, tag2id, id2word, id2tag):

    utterances = list()
    tags = list()
    starts = list()
    startid = list()

    utt_count = 0
    temp_startid = 0

    for line in open(testFile, 'r'):
        d = line.split("\t")
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

        for i in xrange(len(mywords)):
            if mywords[i] not in word2id:
                temp_utt.append(1)
            else:
                temp_utt.append(word2id[mywords[i]])
            if mytags[i] not in tag2id:
                temp_tags.append(1)
            else:
                temp_tags.append(tag2id[mytags[i]])

        utt_count +=1
        utterances.append(temp_utt)
        tags.append(temp_tags)
        wordVocabSize = len(word2id)

    data = {'start':starts, 'startid':startid, 'utterances':iutterances, 'tags':tags, 'uttCount':utt_count, 'wordVocabSize':word_vocab_index, 'id2word':id2word, 'id2tag':id2tag}
    return data

def readNum(numFile):
    numList  = map(int, file(numFile).read().strip().split())
    totalList = list()
    cur = 0
    for num in numList:
        cur += num+1
        totalList.append(cur)
    return numList, totalList
