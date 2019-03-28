#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计各词在各种文档中出现的频率
先将rate为同一个level的text连接起来,并统计在这个文档中出现的频数.最后统计所有词出现频数
@author: tide
"""
import pandas as pd
from nltk.book import FreqDist
import json
import numpy as np

data=pd.read_csv('../data/preProcess/labeledCharTrain.csv')

'''
1.统计词在各类文本中分布的频数,并将结果保存
'''
def word_frenquency_static(data,rate): #统计每一个rate的词分布
   texts=list(data[data['rate'] == rate]['review']) #选取同一个rate的评论
   wordlist=[]
   for text in texts:
     text=[word for word in text.lower().strip().split()]
     wordlist.extend(text)
   fdist=FreqDist(wordlist)
   return fdist

rate_unique=data['rate'].unique()
rate_unique.sort()
rate_unique=list(rate_unique)

#统计每个rate下,每种词出现的频数,以及all
for i in (rate_unique):
    path="../data/wordFrequency/word_"+str(i)+".json"
    with open(path, "w", encoding="utf-8") as f:
            json.dump(word_frenquency_static(data,i), f)  

#word_all 所有词的        
texts=list(data['review']) #选取同一个rate的评论
wordlist=[]
for text in texts:
      text=[word for word in text.lower().strip().split()]
      wordlist.extend(text)
fdist_all=FreqDist(wordlist)
path="../data/wordFrequency/word_all.json"
with open(path, "w", encoding="utf-8") as f:
            json.dump(fdist_all, f)  

#打开尝试调用一下
f=open(path,'r')
a=f.read()
dict_fdist_all=eval(a)
print('ok',dict_fdist_all['ok'])


'''
2.计算词在各类文本中的分布概率,并保存结果
'''

#将八个词典保存在一个name_dict词典中
name_dict={}
for i in rate_unique:
    path="../data/wordFrequency/word_"+str(i)+".json"
    f=open(path,'r')
    word_f=eval(f.read())
    name_dict['word_'+str(i)]=word_f

#总词典
path_all="../data/wordFrequency/word_all.json"
f=open(path_all,'r')
word_all=eval(f.read())

def word_distribution(word):    #统计词在各类文档中分布概率,如果没有则认定为均匀分布
    word_dist=np.zeros(8)
    try:
        allFreq=word_all[word]
    except:
        return np.ones(8)/8
    for index,i in enumerate (rate_unique):
        try:
            Dict='word_'+str(i)
            word_dist[index]=name_dict[Dict][word]/allFreq
        except:
            word_dist[index]=0
    return word_dist.tolist()

wordUnique=set()
for word in wordlist:
    if word not in wordUnique:
        wordUnique.add(word)

wordFreq={}
for word in wordUnique:
    wordFreq.update({word:word_distribution(word)})
    
path="../data/wordFrequency/wordFreq.json"
with open(path, "w", encoding="utf-8") as f:
            json.dump(wordFreq, f)   



'''
3.检查是否已经存储的json文件是否能够调用
'''
f = open(path,'r')
a = f.read()
dict_name = eval(a)
f.close()
print('ok在各类文本中的分布信息:',dict_name['ok'])
print('happy在各类文本中的分布信息:',dict_name['happy'])


