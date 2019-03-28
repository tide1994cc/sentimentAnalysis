****
这里主要介绍了,每个文件夹下,各存放的各种文件和数据
***

# 1.raw_data
 raw_data为原始数据文件,下面总共包括三个tsv格式的文件,分别为     
 labelTrainData.tsv,testData.tsv,unlabeledTrainData.tsv
  其中labelTrainData.tsv 信息内容为 id,sentiment(1,0),review,rate(情感极性,1-10)


#2. preprocess
  为raw_data预处理完了以后的数据,三个csv文件 labelchatTrain,testData.csv,wordEmbding
  wordingEmbding是由labelTrainData.tsv+unlabeledTrainData.tsv 而来,用作词向量训练

#3. wordFrenquency
   存储每一个rate文本中,每一个词的词频统计情况,
   有9个文件,分别为word_1.json~word_4.json,word_7.json~word_10.json,外加一个所有单词的词频统计.word_all.json
   wordSenti.json存储的是从sentiwordnet抽取出来的情感词信息.
   wordFreq.json存储的是词的概率分布信息

#4.word2vec
   用来存放训练好的词典 word2Vec.bin

#5.stopword
   停用词词典
#6.wordJson
来自于训练时候产生的文件,为了便于快速查找