#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从sentiwordnet中抽取词的情感信息
@author: tide
"""

from nltk.corpus import sentiwordnet as swn
import numpy as np
import pandas as pd
import json

'''
1.导入所有的词,label,test
'''
label=pd.read_csv('../data/preProcess/labeledCharTrain.csv')
test=pd.read_csv('../data/preProcess/testData.csv')

label_texts=label['review'].tolist()
test_texts=test['review'].tolist()
text_all=label_texts+test_texts

for i in range(len(text_all)):
    text_all[i]=text_all[i].strip().split()

allWords = [word for text in text_all for word in text]

wordUnique=set()
for word in allWords:
    if word not in wordUnique:
        wordUnique.add(word)

'''
2.从sentiwordnet中抽取词语的情感信息并且保存
'''

def score_plt(scores):
    plt=np.zeros(17)
    if len(scores)!=0:
      for score in scores:
          level=int(8*(score+1))
          plt[level]=plt[level]+1
      plt=plt/sum(plt)
    else:
        plt=np.ones(17)/17
    return plt.tolist()
    

def word_to_senti(word):
        wordSet=list(swn.senti_synsets(word))
        length=len(wordSet)
        if(length>=0):  
            scores=[]
            for i in range(length):
                score=wordSet[i].pos_score()-wordSet[i].neg_score()
                if score!=0:
                    scores.append(score)
            scores=score_plt(scores)
        return scores       

        
wordSenti={}
for word in wordUnique:
    wordSenti.update({word:word_to_senti(word)})


'''
3.检查是否能够调用
'''
path="../data/wordFrequency/word_senti.json"
with open(path, "w", encoding="utf-8") as f:
            json.dump(wordSenti, f)   
#调用已经存储的json文件
f = open(path,'r')
a = f.read()
wordSenti = eval(a)
f.close()
print('ok的情感信息为:',wordSenti['ok'])
print('happy的情感信息为:',wordSenti['happy'])


'''
4.将词典信息与词的分布信息合并
'''
with open('../data/wordFrequency/wordFreq.json') as f:
     tem=f.read()
     wordFreq=eval(tem)
print('ok:',wordFreq['ok'])

for word in wordUnique:
    if word in wordFreq:
        freq=wordFreq[word]
    else:
        freq=np.ones(8)/8
    












    