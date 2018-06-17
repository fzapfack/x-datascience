from pyspark import SparkContext
import loadFiles as lf
import numpy as np
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from functools import partial
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

from  pyspark.mllib.classification import SVMWithSGD
from  pyspark.mllib.classification import LogisticRegressionWithLBFGS
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score

import string
import nltk
from nltk.stem.porter import *
from nltk.util import ngrams
from nltk.corpus import stopwords

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.linalg import Vectors


#from pyspark.ml.tuning import CrossValidator
sc = SparkContext(appName="App ")


def doc2words(doc):
	punctuation = set(string.punctuation)
	for w in ['.',',','--',':','!','?','(',')']:
		doc=doc.replace(w,' ')
	words = doc.strip().split(' ')
	words = [w.lower() for w in words if w not in punctuation]
	stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
				'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', "isn't",'are',"aren't",
                'was', "wasn't", 'were', "weren't", 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now','films',
                'film','movies','movie','br',' ', '']
	words = [w for w in words if w not in stop_words]
	words = [w for w in words if not any(i in w for i in ['/','<','>'])]
	return words


def FinalPredict(name_text,dictionary,model):
	#words=name_text[1].strip().split(' ')
	words=doc2words(name_text[1])
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text[0], model.predict(SparseVector(len(dictionary),vector_dict)))



data,Y=lf.loadLabeled("/Users/sofia/Desktop/big-data-project/data/train")
print "data loaded, length {}".format(len(data))
dataRDD=sc.parallelize(data,numSlices=16)
lists=dataRDD.map(doc2words)
#lists=dataRDD.map(doc2words).collect()

# create dict
all=[]
for l in lists.collect():
	all.extend(l)
dict=set(all)

# TF-IDF
hashingTF = HashingTF(numFeatures=len(dict))
tf = hashingTF.transform(lists)
tf.cache()
idf = IDF().fit(tf)
tfidf = idf.transform(tf).collect()
#data2 =tfidf.collect()

# a = lists.collect()
# b = tfidf.collect()
# print "type tfidf {} len {}".format(type(b),len(b))
# c=b[0]
# print "type c {} len {}".format(type(c),len(c))




## cross-validaton/Grid-search:
cv = ShuffleSplit(len(tfidf), n_iter=3, test_size=0.3, random_state=42)

nb=NaiveBayes()
lr=LogisticRegressionWithLBFGS()
svm=SVMWithSGD()
models = [lr, nb, svm]
scores = {model.__name__: [] for model in models}
grids = [ParamGridBuilder()\
.baseOn({lr.labelCol: 'l'})\
.baseOn([lr.predictionCol, 'p'])\
.addGrid(lr.regParam, [1.0, 2.0]) \
.addGrid(lr.maxIter, [0, 1]).build(),\
ParamGridBuilder()\
.addGrid(nb.lambda_, [0, 1]) \
.addGrid(nb.maxIter, [0, 1]).build() ,\
ParamGridBuilder()\
.baseOn({svm.labelCol: 'l'})\
.baseOn([svm.predictionCol, 'p'])\
.addGrid(svm.regParam, [1.0, 2.0]) \
.addGrid(svm.maxIter, [0, 1]).build()
]

	#### Model
	
for model,grid in models,grids:
		#### Model training with grid search
		
		evaluator = BinaryClassificationEvaluator()
		cv = CrossValidator(estimator=model, estimatorParamMaps=grid, evaluator=evaluator)
		model_trained = cv.fit(tfidf)
		evaluator.evaluate(model_trained.transform(tfidf))
		
		
		#mb=sc.broadcast(model_trained)


		### Model testing
		
		print "Accuracy %.5f" %score

for key, value in scores.items():
	print "%s : mean=%.5f, std=%.5f " %(key, np.mean(value), np.std(value))
	# for i,model in enumerate(models):
	# 	print " %s mean accuracy_score : " accuracy_score(Y_test,Y_pred)








# #Train NaiveBayes
# model=NaiveBayes.train(labeledRDD)
# #broadcast the model
# mb=sc.broadcast(model)

# test,names=lf.loadUknown('./data/test')
# name_text=zip(names,test)
# #for each doc :(name,text):
# #apply the model on the vector representation of the text
# #return the name and the class
# predictions=sc.parallelize(name_text).map(partial(Predict,dictionary=dict_broad.value,model=mb.value)).collect()

# output=file('./classifications.txt','w')
# for x in predictions:
# 	output.write('%s\t%d\n'%x)
# output.close()

# lr=NaiveBayes()
# cv = CrossValidator(estimator=lr,numFolds=3)
# cvModel = cv.fit(labeledRDD)
# evaluator.evaluate(cvModel.transform(labeledRDD))

