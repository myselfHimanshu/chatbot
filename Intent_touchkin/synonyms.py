import pandas as pd
from correction import correct_

#text = "i'll there for you"
#print(correct_(text))

sentences_file = "cleaned_sentences.csv"
vocab_file = "Vocabulary.csv"

sentences_df = pd.read_csv(sentences_file)
vocab_df = pd.read_csv(vocab_file)

vocab_columns = vocab_df.columns

words2col = {}
for column in vocab_columns[:-1]:
    if "Unnamed" in column:
        continue
    words_ = set(vocab_df[column].tolist())
    for word in words_:
        if word=="nan":
            continue
        elif word not in words2col:
            words2col[str(word).strip()] = column.strip()
            
    
sentences = sentences_df["sentences"].tolist()    
tagged_sentences = []
cleaned_sentences = []

for sentence in sentences:
    clean_ = correct_(sentence).strip()
    cleaned_sentences.append(clean_)
    tag_sentence = ""
    for word in clean_.split():
        if word in words2col:
            tag_sentence+=words2col[word]+" "
        else:
            tag_sentence+="0 "
    tagged_sentences.append(tag_sentence.strip())
    

    

        
            
    
            
