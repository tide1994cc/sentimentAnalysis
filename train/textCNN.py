# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 18:26:36 2019

@author: tide1
"""
import os
import csv
import time
import datetime
import random
import json

from collections import Counter
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

'''
1. 配置参数
'''

class TrainingConfig(object):  #训练参数
    epoches = 10 
    evaluateEvery = 100  
    checkpointEvery = 100
    learningRate = 0.001
    
class ModelConfig(object):  #CNN模型参数
    embeddingSize = 317 #向量长度=词向量维度+wordSF维度
    numFilters = 128
    filterSizes = [2, 3, 4, 5]  #卷积大小
    dropoutKeepProb = 0.5
    l2RegLambda = 0.0
   
    
class Config(object):          #数据预处理参数
    sequenceLength = 200  # 一个文本只选取前200个词，没有那么长久贴上pad
    batchSize = 128
    dataPath = "../data/preProcess/labeledCharTrain.csv"
    stopWordPath = "../data/stopword"
    numClasses = 2
    rate = 0.8  # 训练集的比例
    training = TrainingConfig()
    model = ModelConfig()
    word2VecPath='../data/word2vec/word2Vec.bin' #使用自己训练的词库
    #word2VecPath='../data/GoogleNewsVec.bin'# 使用公开google词库
   
# 实例化配置参数对象
config = Config()
'''
# 2.数据预处理的类，生成训练集和测试集
'''
class Dataset(object):
    def __init__(self, config): #1
        self._dataSource = config.dataPath #数据存放路径
        self._stopWordSource = config.stopWordPath  #停用词存放路径
        self._word2VecSource=config.word2VecPath
        
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize #词向量维度
        self._batchSize = config.batchSize #
        self._rate = config.rate  #训练集的百分比
        
        self._stopWordDict = {} #停用词词典
        
        self.trainReviews = [] #20000*400的矩阵 ,只记录400个词,构建等长的词向量
        self.trainLabels = []  #20000*1的label        
        self.evalReviews = []   #评价 5000*400
        self.evalLabels = []   #5000*1
        
        self.wordEmbedding =None  
        
        self._wordToIndex = {}
        self._indexToWord = {}
        
    def _readData(self, filePath): #2
        """
        从csv文件中读取数据集,返回reviews和label
        """        
        df = pd.read_csv(filePath)
        labels = df["sentiment"].tolist() #转为list
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review] #对每一个review进行分词,形成list

        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, wordToIndex): #文本向量化,构建向量模型
        """
        利用之前建立好的词典，将数据集中的每条评论用index表示.
        """   
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength       
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)
            
        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec

    def _genTrainEvalData(self, x, y, rate): #4
        """
        生成训练集和验证集
        """       
        reviews = []
        labels = []
        
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._wordToIndex)
            reviews.append(reviewVec)         
            labels.append([y[i]])
            
        trainIndex = int(len(x) * rate)     
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")     
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels
        
    def _genVocabulary(self, reviews): #5
        """
        生成json文件，用来存储 index和对应的word
        获得每一个词的词向量，构成词向量矩阵
        """        
        allWords = [word for review in reviews for word in review] #文本所有的词        
        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]        
        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True) #按照词频排序
        
        # 去除低频词,词频低于5的进行移除
        words = [item[0] for item in sortWordCount if item[1] >= 5]
        
        #得到3w多个词以及其相对应的词向量
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
         
        #构建词典
        self._wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToWord = dict(zip(list(range(len(vocab))), vocab))       
        with open("../data/wordJson/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._wordToIndex, f)    
        with open("../data/wordJson/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToWord, f)
            
    def _getWordEmbedding(self, words): #返回一个每一个词的,word2vec向量
        """
        返回能在词典中找到的词，
        返回词向量矩阵
        输出不在矩阵中的词
        """      
        #word2vec
        wordVec = gensim.models.KeyedVectors.load_word2vec_format(self._word2VecSource, binary=True)
        
        #wordSenti
        with open('../data/wordFrequency/word_senti.json') as f:
             tem=f.read()
             wordFreq=eval(tem)
        
        vocab = []
        wordEmbedding = []       
        # 添加 "pad" 和 "UNK", 
        vocab.append("pad") #没有那么长
        vocab.append("UNK")  #查不到的词
        wordEmbedding.append(np.zeros(self._embeddingSize))#第一行为零,对应pad
        wordEmbedding.append(np.random.randn(self._embeddingSize)) #第二行随机 对应查不到
        
        for word in words:
            try:
                vector_1 = wordVec.wv[word]
                vector_2 = np.array(wordFreq[word])
                vector=np.concatenate([vector_1,vector_2])
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
                
        return vocab, np.array(wordEmbedding)
    
    def _readStopWord(self, stopWordPath): #7
        """
        读取停用词
        """      
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
            
    def dataGen(self):
        # 初始化停用词
        self._readStopWord(self._stopWordSource)
        
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        
        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)
        
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
              
data = Dataset(config)
data.dataGen()
 
'''
3. 输出batch数据集
'''
def nextBatch(x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm) #数组从新排序
        x = x[perm]
        y = y[perm]      
        numBatches = len(x) // batchSize #总共的批次

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = np.array(x[start: end], dtype="int64")
            batchY = np.array(y[start: end], dtype="float32")          
            yield batchX, batchY #每次都可以返回，但不会终止程序运行
'''
# 4.构建模型
'''
tf.reset_default_graph()
class TextCNN(object):
    """
    Text CNN 用于文本分类
    """
    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb") #用来防止过拟合
        
        # 定义l2损失
        l2Loss = tf.constant(0.0) #创建一个常数张量,传入list或者数值来填充
        
        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W") #tf.cast,类型转换函数 将其转换成 float类型
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.inputX) #tf.nn.embedding_lookup 寻找的embedding data中的对应的行下的vector。
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            self.embeddedWordsExpanded = tf.expand_dims(self.embeddedWords, -1)

        # 创建卷积和池化层
        pooledOutputs = []
        #多通道技术卷积技术，这样会使生成的feature map 特征具有多样性
        # 有三种size的filter，2，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filterSize in enumerate(config.model.filterSizes):
            with tf.name_scope("conv-maxpool-%s" % filterSize):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                filterShape = [filterSize, config.model.embeddingSize, 1, config.model.numFilters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[config.model.numFilters]), name="b")
                conv = tf.nn.conv2d(
                    self.embeddedWordsExpanded, #input
                    W,         #卷积核[卷积核高度，滤波器宽度，图像通道数，滤波器个数]
                    strides=[1, 1, 1, 1],
                    padding="VALID", #边缘不填充
                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, config.sequenceLength - filterSize + 1, 1, 1],  # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooledOutputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中
        # 得到CNN网络的输出长度
        numFiltersTotal = config.model.numFilters * len(config.model.filterSizes)
        
        # 池化后的维度不变，按照最后的维度channel来concat
        #合并结果为[batch,14,14,4*channel]的数据
        self.hPool = tf.concat(pooledOutputs, 3)
        
        # 摊平成二维的数据输入到全连接层
        self.hPoolFlat = tf.reshape(self.hPool, [-1, numFiltersTotal])

        # dropout
        with tf.name_scope("dropout"): 
            self.hDrop = tf.nn.dropout(self.hPoolFlat, self.dropoutKeepProb)
       
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[numFiltersTotal, 1],
                initializer=tf.contrib.layers.xavier_initializer())
            outputB= tf.Variable(tf.constant(0.1, shape=[1]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.predictions = tf.nn.xw_plus_b(self.hDrop, outputW, outputB, name="predictions")
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.5), tf.float32, name="binaryPreds")
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):        
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
            
            
'''            
5.定义性能指标函数
'''
def mean(item):
    return sum(item) / len(item)

def genMetrics(trueY, predY, binaryPredY):
    #生成acc和auc值
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)
    
    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)
'''
# 6.训练模型
'''
# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels
wordEmbedding = data.wordEmbedding

'''
# 7.定义计算图
'''
with tf.Graph().as_default():
    
    #配置session的运行方式，使用gpu运行
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False) 
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  
    sess = tf.Session(config=session_conf)
    
    # 定义会话
    with sess.as_default():
        cnn = TextCNN(config, wordEmbedding)
        
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 梯度修剪主要避免训练梯度爆炸和消失问题
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)  #直方图 一般用来显示训练过程中变量的分布情况
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g)) #多用来画loss
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys")) 
        print("Writing to {}\n".format(outDir))
        
        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()
        
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
        
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5) 
        
        # 保存模型的一种方式，保存为pb文件
        builder = tf.saved_model.builder.SavedModelBuilder("../model/textCNN/savedModel")
        sess.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            feed_dict = {
              cnn.inputX: batchX,
              cnn.inputY: batchY,
              cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
          #  timeStr = datetime.datetime.now().isoformat()
           # acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            #print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc, auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)

        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
              cnn.inputX: batchX,
              cnn.inputY: batchY,
              cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            
            evalSummaryWriter.add_summary(summary, step)
            
            return loss, acc, auc, precision, recall
       #####
       #开始训练
       #####
        for i in range(config.training.epoches):
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])
                currentStep = tf.train.global_step(sess, globalStep) 
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)                
                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str, currentStep, mean(losses), 
                                                                                                       mean(accs), mean(aucs), mean(precisions),
                                                                                                       mean(recalls)))                
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/textCNN/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
                    
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

        outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()