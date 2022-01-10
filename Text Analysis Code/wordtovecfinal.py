import logging
import pandas as pd
import numpy as np
import gensim
import nltk
import re
from bs4 import BeautifulSoup



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




 
#from itertools import islice


wv = gensim.models.KeyedVectors.load_word2vec_format('D:/data/word_vec/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin', binary=True)
wv.init_sims(replace=True) 


from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.stats import gmean

def word_averaging(wv, words):
    all_words, mean = set(), []
    cou=0
    for word in words:
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.syn0norm[wv.vocab[word].index])
            all_words.add(wv.vocab[word].index)

    if not mean:
        logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
        return np.zeros(wv.vector_size,)
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    #mean = gensim.matutils.unitvec(gmean(np.array(mean))).astype(np.float32)
    #mean = gensim.matutils.unitvec(np.percentile(np.array(mean),25,axis=0)).astype(np.float32)
    #mean = gensim.matutils.unitvec(skew(np.array(mean))).astype(np.float32)
   #mean = gensim.matutils.unitvec(np.percentile(np.array(mean),50,axis=0)).astype(np.float32)
    return mean


def  word_averaging_list(wv, text_list):
    return np.vstack([word_averaging(wv, post) for post in text_list ])

def w2v_tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text, language='english'):
        for word in nltk.word_tokenize(sent, language='english'):
            tokens.append(word)
    return tokens

fn=['JDT','PDE','CDT','Thunderbird','Bugzilla','Platform']
for i in range(0,6):
    fname='D:/data/msr2013-bug_dataset-master/'+fn[i]+'.csv'
    df = pd.read_csv(fname,encoding='latin-1')
    df['Var3'] = df['Var3'].apply(clean_text)
    print(df.head(10))
    comtdata=df
    test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r['Var3']), axis=1).values
    X_comtdata_average1 = word_averaging_list(wv,test_tokenized)
    fname='D:/data/msr2013-bug_dataset-master/w2vec'+fn[i]+'.csv'
    np.savetxt(fname,X_comtdata_average1, delimiter=',', fmt='%f')


#df = pd.read_csv('D:/data/bug-triagingbug-assignment/data.csv',encoding='latin-1')
#df['Summary'] = df['Summary'].apply(clean_text)
#print(df.head(10))

#comtdata=df

#test_tokenized = comtdata.apply(lambda r: w2v_tokenize_text(r['Summary']), axis=1).values

#X_comtdata_average1 = word_averaging_list(wv,test_tokenized)
#np.savetxt('D:/data/bug-triagingbug-assignment/w2vdata.csv',X_comtdata_average1, delimiter=',', fmt='%f')


