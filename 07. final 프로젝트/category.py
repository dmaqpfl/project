import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from konlpy.tag import Kkma, Hannanum, Komoran, Mecab
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


from numpy.linalg import norm
from collections import Counter

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

import os
import re

from sklearn.model_selection import train_test_split

import tensorflow as tf


stopwords1 = pd.read_csv('stopwords.txt')
stopwords2 = pd.read_csv('stopwords_add.txt')

cate_dic = {0 : '부동산분양정보', 1 : '금융', 2 : '기업일반정보',3 :'증권,주식',4:'기업기술정보',5:'기업실적등', 
           6: '금융정책', 7:'부동산정책', 8:'가상화폐', 9: '주식시장'}

mecab = Mecab()

s_list1 = []
for i in stopwords1.iloc[:,0]:
    s_list1.append(i)
s_list2 = []
for i in stopwords2.iloc[:,0]:
    s_list2.append(i)
        
st = s_list1 + s_list2


# 텍스트 클린
def text_clean(doc):
    doc =re.sub("[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣 ]", "", doc)
    return doc

# 1글자 및 리스트로 전환하는 함수(불용어처리 포함)
def text_preprocession(x) :
    
    x = mecab.nouns(x) # 매켑 명사 추출 후 텍스트를 리스트화
    
  
    test_li = []

    # 1글자 및 불용어 제거
    for i in x:
        if (len(i) > 1) & (i not in st) :
            test_li.append(i)
            
    return test_li


# 전처리 후 토크나이징, 그리고 패딩하는 함수
def text_pre(x):
     
    # 기사를 토크나이징
    tok = Tokenizer()
    tok.fit_on_texts(x)
    
    threshold = 0 #단어빈도 몇개 이하를 버릴건지 결정해주는 변수.
    total_cnt = len(tok.word_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도 수가 threshold보다 작은 단어의 등장 빈도 수의 총 합


    #단어와 빈도수의 쌍(pair)을 key와 value로 받는다.

    for key, value in tok.word_counts.items():
        total_freq = total_freq + value
        if(value < threshold):
            rare_cnt = rare_cnt +1
            rare_freq = rare_freq+value
            
    vocab_size = total_cnt - rare_cnt + 1 
    tok = Tokenizer(vocab_size) 
    tok.fit_on_texts(x)
    x = tok.texts_to_sequences(x)
    
    max_len = 1
    x = pad_sequences(x, maxlen = max_len)
    
    return x


# 패딩된 array에서 카테고리화 하는 함수
def categorizer(text):
     # 모델을 불러오기
    model = load_model('best_model_220103_R3.h5')
    
    # 모델 적용
    yhat = model.predict(text)
    
    # 기사가 1개이기 때문에 1행만 출력
    y = pd.DataFrame(yhat) #.iloc[0,:]

    y.columns = cate_dic.values()
    
     #최대값인 칼럼 출력
 
    li = []

    for i in y:
        li.append(y[i].max())    
    
    dd = {}
    j = 0
    while j < len(y.columns):
        dd[y.columns[j]] = li[j]
        j += 1
    
    aaa = pd.DataFrame(columns = ['categ','score'])
    aaa['categ'] = dd.keys()
    aaa['score'] = dd.values()
    aaaa = str(aaa[aaa['score'] == aaa['score'].max()]['categ'])
    
    return re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", aaaa).strip()