***
基于情感词典+词嵌入模型,利用CNN进行的文本情感分析
***
基于word2vec只考虑到了词的上下文关系,并没有考虑到所处分类和词本身的情感色彩.  
本项目在基于此等考虑,引入情感词典sentiwordnet,并且对词在各类文本分布情况进行了分析.
实验结果证明了,改进是相当有效的.



***
1.file
***
#1.data
数据存放在data中,其中有data_readme.md,介绍了各种数据的由来,每个文件夹下的数据所含有的信息

#2.preprocess
预处理文件夹,大致的readme.md大致介绍了数据预处理过程,每个py文件所起的作用

#3.train
模型训练文件夹

#model
存放textCNN训练的模型,在训练后会出现



***
2.requirements: 
***
nltk
tensorflow-gpu
json
numpy
pandas 
gensim


***
appendix
***
本项目是使用的词向量是自训练的词向量,
googleNewsVector太大,解压后3个多g,
需要的同学可以发邮件给我,tide1994cc@163.com
下载后存放在data/word2Vec中,
修改textCNN里面 config.word2Vecpath参数及可使用












