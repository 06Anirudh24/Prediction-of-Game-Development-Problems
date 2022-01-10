
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
import numpy as np
import gensim 
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 


sample = open("D:/data/aman_api_postmatem/alice.txt", "r",encoding="utf8") 
s = sample.read() 
f = s.replace("\n", " ") 

data = [] 
for i in sent_tokenize(f): 
	temp = [] 
	
	# tokenize the sentence into words 
	for j in word_tokenize(i): 
		temp.append(j.lower()) 

	data.append(temp) 
model1 = gensim.models.Word2Vec(data, min_count = 1, 
							size = 300, window = 5) 


model2 = gensim.models.Word2Vec(data, min_count = 1, size = 300, 
											window = 5, sg = 1) 



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text








def wordprint(words,model1):
    all_words, mean = set(), []
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in model1.wv.vocab:
            a=np.array(model1[word])
            mean.append(a)
    if not mean:
        # FIXME: remove these examples in pre-processing
        return np.zeros(100,)   
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def  word_averaging_listn(wv, text_list):
    return np.vstack([wordprint(post,wv) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens

import numpy as np
fn=['dataset']
for i in range(0,1):
    fname='D:/data/aman_api_postmatem/'+fn[i]+'.csv'
    df = pd.read_csv(fname,encoding='latin-1')
    df['quote'] = df['quote'].apply(clean_text)
    print(df.head(10))
    comtdata=df
    test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r['quote']), axis=1).values
    X_comtdata_average1 = word_averaging_listn(model1,test_tokenized)
    fname='D:/data/aman_api_postmatem/cbow'+fn[i]+'.csv'
    np.savetxt(fname,X_comtdata_average1, delimiter=',', fmt='%f')
    X_comtdata_average1 = word_averaging_listn(model2,test_tokenized)
    fname='D:/data/aman_api_postmatem/skg'+fn[i]+'.csv'
    np.savetxt(fname,X_comtdata_average1, delimiter=',', fmt='%f')


#df = pd.read_csv('D:/data/bug-triagingbug-assignment/data.csv',encoding='latin-1')
#df['Summary'] = df['Summary'].apply(clean_text)
#print(df.head(10))
#comtdata=df

#test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r['Summary']), axis=1).values

#X_comtdata_average1 = word_averaging_listn(model1,test_tokenized)
#np.savetxt('D:/data/bug-triagingbug-assignment/cbow.csv',X_comtdata_average1, delimiter=',', fmt='%f')

#X_comtdata_average1 = word_averaging_listn(model2,test_tokenized)
#np.savetxt('D:/data/bug-triagingbug-assignment/sg.csv',X_comtdata_average1, delimiter=',', fmt='%f')

