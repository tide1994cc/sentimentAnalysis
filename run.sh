#mkdir
cd data
mkdir preProcess
mkdir word2vec
mkdir wordFrequency
mkdir wordJson



# data preparing
export PATH=/home/tide/anaconda3/bin:$PATH 

cd ..
cd preprocess
python processData.py
python word2vec_gen.py
python wordFrequency.py
python sentiword.py
python mergeSentiFreq.py



#train
cd ..
cd train
python textCNN.py