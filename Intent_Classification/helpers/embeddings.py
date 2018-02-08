import re
import numpy as np

class PredefinedEmbedding():
    def __init__(self, embeddingFile):
        self.embeddings = readEmbeddings(embeddingFile)

    def getEmbeddingDim(self):
        return self.embeddings["embeddingSize"]

    def getWordEmbeddings(self, word):
        if word == "BOS":
            return self.embeddings["embeddings"]["<\s>"]
        elif word == "EOS":
            return self.embeddings["embeddings"]["<\s>"]
        elif word in self.embeddings["embeddings"]:
            return self.embeddings["embeddings"][word]
        else:
            return np.zeros((self.embeddigs['embeddingSize'],1))





def readEmbeddings(embeddingFile):
    wordEmbeddings = {}
    first = True
    p = re.compile('\s+')

    for line in open(embeddingFile, 'r'):
        d = p.split(line.strip())
        if(first):
            first = False
            size = len(d)-1
        else:
            if(size!=len(d)-1):
                print("Problem with embedding file, not all vectors are of same length")
                exit()
        currentWord = d[0]
        wordEmbeddings[currentWord] = np.zeros((size,1))

        for i in range(1, len(d)):
            wordEmbeddings[currentWord][i-1] = float(d[i])
    embeddings = {'embeddings':wordEmbeddings, 'embeddingSize':size}
    return embeddings
