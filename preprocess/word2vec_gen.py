import gensim
import logging
from gensim.models import word2vec
'''
1.数据读取
'''

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, 
 #PathLineSentences等。
sentences = word2vec.LineSentence("../data/preProcess/wordEmbdiing.txt")
'''
2.训练
'''
# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用cbow模型，模型保存为bin格式,负采样
model = gensim.models.Word2Vec(sentences, size=300, sg=0, iter=8, negative=16,min_count=3)  
model.wv.save_word2vec_format("../data/word2vec/word2Vec" + ".bin", binary=True) 

# 加载bin格式的模型
wordVec = gensim.models.KeyedVectors.load_word2vec_format("../data/word2vec/word2Vec.bin", binary=True)

word_vec=wordVec['man']









