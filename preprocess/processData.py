import pandas as pd
from bs4 import BeautifulSoup
import re

with open("../data/raw_data/unlabeledTrainData.tsv", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]    

with open("../data/raw_data/labeledTrainData.tsv", "r") as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

with open("../data/raw_data/testData.tsv", "r") as f:
     testData = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]    



unlabel = pd.DataFrame(unlabeledTrain[1: ], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1: ], columns=labeledTrain[0])
test= pd.DataFrame(testData[1: ], columns=testData[0])

unlabel.head(5)
label.head(5)
test.head(5)


#获得已经标记的评论的评分信息
def getRate(subject):
    splitList = subject[1:-1].split("_")
    return int(splitList[1])
label["rate"] = label["id"].apply(getRate)
test["rate"] = test["id"].apply(getRate)
label.head(5)

#文本清理
def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject =re.sub("[＞＠［＼］-＾《》\"「」『』【】〖〗〘〙〚〛〜〝/:;<=>?@[\\]$^_`<>,!'%&)-/*+(#$.{|}~]+", "", newSubject)
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)    
    return newSubject
    
unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)
unlabel["review"] = unlabel["review"].apply(cleanReview)
test["review"] = test["review"].apply(cleanReview)
label.head(5)

def sentiment(rate): #获取test的0-1二分类
    rate=int(rate)
    if rate<=5:
        return 0
    else:
        return 1

test['sentiment']=test['rate'].apply(sentiment)

newDf = pd.concat([unlabel["review"], label["review"]], axis=0) 
newDf.to_csv("../data/preProcess/wordEmbdiing.txt", index=False) #用作训练词向量

newLabel = label[["review", "sentiment", "rate"]]
newLabel.to_csv("../data/preProcess/labeledCharTrain.csv", index=False) #训练集存储

newtest=test[["review", "sentiment", "rate"]]
newtest.to_csv("../data/preProcess/testData.csv", index=False) #测试集存储
