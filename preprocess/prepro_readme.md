***
这个文件夹主要存放了预处理文件,处理顺序为1,2,3,4,5
***

1.processData
对最原始的数据进行处理,从tsv转为csv,文本清洗等工作


2.word2vec.py
词向量训练,保存为word2Vec.bin


3.wordFrequency.py
对labeldata,每一个rate下的词频进行统计,并生成word_rate.json文件存放起来
然后计算词的概率分布信息,并保存起来,保存为wordFreq.json


4.sentiword.py
从sentiwordnet库中抽取词语的情感信息,保存为wordSenti.json


5.mergeSentiFreq.py
将3,4部生成的词的信息,
按照一定的规则进行合并,保存为wordSentiFreq.json