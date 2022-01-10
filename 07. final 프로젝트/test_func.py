import pandas as pd
import numpy as np
from konlpy.tag import Kkma, Okt

kkma = Kkma()
okt = Okt()

import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tfidf = TfidfVectorizer()
cnt_vec = CountVectorizer()

from sklearn.preprocessing import normalize

stopwords = pd.read_csv('stopwords.txt',header=None)
stopwords.columns = ['word']


def text2sentences(text) : 
    sentences = kkma.sentences(text)
    if len(sentences) <= 10 :
        return False
    else :
        return sentences

def get_nouns(sentences) :
    nouns = []
    for sentence in sentences :
        sentence = text_clean(sentence)
        if sentence is not '' :
            nouns.append(' '.join([noun for noun in okt.nouns(str(sentence))
                                  if noun not in stopwords and len(noun) > 1]))
    return nouns

def build_sent_graph(sentence) :
    tfidf_mat = tfidf.fit_transform(sentence).toarray()
    graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
    return graph_sentence

def bulid_word_graph(sentence) : 
    cnt_vec_mat = normalize(cnt_vec.fit_transform(sentence).toarray().astype(float), axis=1)
    vocab = cnt_vec.vocabulary_
    return np.dot(cnt_vec_mat, cnt_vec_mat.T), {vocab[word] : word for word in vocab}

###########################

def get_rank(graph, d=0.85) :
    A = graph
    matrix_size = A.shape[0]
    for id in range(matrix_size) : 
        A[id,id] = 0
        link_sum = np.sum(A[:,id])
        if link_sum != 0 :
            A[:,id] /= link_sum
        A[:,id] *= -d
        A[id,id] = 1
    B = (1-d) * np.ones((matrix_size, 1))
    ranks = np.linalg.solve(A,B)
    return {idx : r[0] for idx, r in enumerate(ranks)}

###########################

def summarize_no(sentences, sorted_sent_rank_idx, sent_num=3) :
    summary = []
    index = []
    for idx in sorted_sent_rank_idx[:sent_num] :
        index.append(idx)
    index.sort()
    
    for idx in index :
        summary.append(sentences[idx])
    texts = ''
    for text in summary :
        texts = ''.join(text)
    return texts

# def key_word(word_num=10) :
#     keywords = []
#     index = []
#     for idx in sorted_word_rank_idx[:word_num] :
#         index.append(idx)
#     for idx in index :
#         keywords.append(idx2word[idx])
#     print(keywords)

    
###########################

def text_clean(doc):
    doc =re.sub("[^a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]", "", doc)
    return doc

def stopwords_del(x) :
    if any(stopwords['word'] == x) :
        return False
    return True
