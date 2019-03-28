#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:00:15 2019
@author: tide
"""
import numpy as np
import json

pathFreq="../data/wordFrequency/wordFreq.json"
f = open(pathFreq,'r')
a = f.read()
wordFreq = eval(a)
f.close()
print('ok在各类文本中的分布信息:',wordFreq['ok'])
print('happy在各类文本中的分布信息:',wordFreq['happy'])


pathSenti="../data/wordFrequency/word_senti.json"
f = open(pathSenti,'r')
a = f.read()
wordSenti = eval(a)
f.close()
print('ok的情感信息为:',wordSenti['ok'])
print('happy的情感信息为:',wordSenti['happy'])


wordSet=set()
for word in wordSenti:
    if word not in wordSet:
        wordSet.add(word)

#定义融合函数
def wordSentiFreq(word):
    wordS=wordSenti[word]
    try:
        wordF=wordFreq[word]
    except:
        wordF=(np.ones(17)/17).tolist()
    for i in range(8):    #wordF的长度 8
        wordS[2*i]=wordS[2*i]+wordF[i]
    return wordS
    
wordSF={}
for word in wordSet:
    wordSF.update({word:wordSentiFreq(word)})
    
path="../data/wordFrequency/wordSF.json"
with open(path, "w", encoding="utf-8") as f:
            json.dump(wordSF, f)   
    

        
    