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

import re
import string
import nltk
from nltk.stem.porter import *
from nltk.util import ngrams
from nltk.corpus import stopwords

#from pyspark.ml.tuning import CrossValidator
sc = SparkContext(appName="App ")
num_slices=16

## Split Text in list of words
def doc2words(doc,stemming=False):
	# Punctuation removal
	doc = re.sub("[^a-zA-Z]"," ", doc)
	# Split in words
	words=doc.strip().split(' ')
	words = [w.lower() for w in words]
	# stop-words based on the previous lab we add nltk stopwords
	stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
					'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
	                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
	                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
	                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
	                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
	                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
	                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
	                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
	                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
	                'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so','review','reviews',
	                'than', 'too', 'very', 's', 't', 'can','could', 'will','would', 'just', 'don', 'should', 'now','films',
	                'film','movies','movie','br','http','also','seem','comment','comments','user','users','imdb','movies', ' ', '']
	stop_words.extend([w.encode('ascii','ignore') for w in stopwords.words("english")])
	stop_words.extend([w for w in words if len(w)<3]) 
	stop_words = set(stop_words)
	bigrams = zip(words,words[1:])
	trigrams = zip(words,words[1:],words[2:])
	bigrams = [" ".join(gram) for gram in bigrams if not any(i in stop_words for i in gram)]
	trigrams = [" ".join(gram) for gram in trigrams if not any(i in stop_words for i in gram)]
	words = [w for w in words if w not in stop_words and w not in punctuation]
	words.extend(bigrams)
	words.extend(trigrams)

	# # Stemming
	if stemming: 
		stemmer = PorterStemmer()
		words = [stemmer.stem(unicode(w, "utf-8")) for w in words] 
	return list(set(words)) # return the words in each document

def doc2vec(doc,dictionary):
	#words=doc_class[0].strip().split(' ')
	words=doc2words(doc)
	#create a binary vector for the document with all the words that appear (0:does not appear,1:appears)
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return SparseVector(len(dictionary),vector_dict)

def createBinaryLabeledPoint(doc_class,dictionary):
	#words=doc_class[0].strip().split(' ')
	words=doc2words(doc_class[0])
	#create a binary vector for the document with all the words that appear (0:does not appear,1:appears)
	#we can set in a dictionary only the indexes of the words that appear
	#and we can use that to build a SparseVector
	vector_dict={}
	for w in words:
		vector_dict[dictionary[w]]=1
	return LabeledPoint(doc_class[1], SparseVector(len(dictionary),vector_dict))

def FinalPredict(name_text,dictionary,model):
	#words=name_text[1].strip().split(' ')
	words=doc2words(name_text[1])
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return (name_text[0], model.predict(SparseVector(len(dictionary),vector_dict)))

def createTestPoint(doc,dictionary):
	words=doc2words(doc)
	vector_dict={}
	for w in words:
		if(w in dictionary):
			vector_dict[dictionary[w]]=1
	return SparseVector(len(dictionary),vector_dict)

# Load reviews data
data,Y=lf.loadLabeled("./data/train")
print "data loaded"

## cross-validaton
cv = ShuffleSplit(len(data), n_iter=3, test_size=0.3, random_state=42)
models = [NaiveBayes, LogisticRegressionWithLBFGS, SVMWithSGD]
scores = {model.__name__: [] for model in models}
for i_cv, i_temp in enumerate(cv):
	i_train = i_temp[0]
	i_test = i_temp[1]
	data_train = [data[i] for i in i_train]
	Y_train = [Y[i] for i in i_train]
	data_test = [data[i] for i in i_test]
	Y_test = [Y[i] for i in i_test]
	print "End Train/test split. Data samples length %d" % (len(data_train))

	dataRDD=sc.parallelize(data_train,numSlices=num_slices)
	#map data to a binary matrix
	#1. get the dictionary of the data
	#The dictionary of each document is a list of UNIQUE(set) words 
	print doc2words(data_train[0])
	lists=dataRDD.map(doc2words).collect()
	all=[]
	#combine all dictionaries together (fastest solution for Python)
	for l in lists:
		all.extend(l)
	dict=list(set(all)) #I think list are faster for indexing

	print "Number of elements in dictionnary %d" % (len(dict))
	#compute the vector of eah word
	dictionary={}
	for word in dict:
		dictionary[word]=i
	#we need the dictionary to be available AS A WHOLE throughout the cluster
	dict_broad=sc.broadcast(dictionary)
	#build labelled Points from data

	### ****** for Svetlana
	## Create a function like createBinaryLabeledPoint where each doc is a column 
	# dcRDD=sc.parallelize(data_train,numSlices=num_slices)
	# doc2vecRDD=dcRDD.map(partial(doc2vec,dictionary=dict_broad.value))
	### do pca of doc2vecRDD
	# train the same way 


	data_class=zip(data_train,Y_train)#if a=[1,2,3] & b=['a','b','c'] then zip(a,b)=[(1,'a'),(2, 'b'), (3, 'c')]
	dcRDD=sc.parallelize(data_class,numSlices=num_slices)
	#get the labelled points
	labeledRDD=dcRDD.map(partial(createBinaryLabeledPoint,dictionary=dict_broad.value))

	testRDD=sc.parallelize(data_test,numSlices=num_slices)


	### ****** for Svetlana
	# Transfor test point in the PCA space 

	testRDD2 = testRDD.map(partial(createTestPoint,dictionary=dict_broad.value))
	#print "End features extraction"

	#### Model
	
	
	for model in models:
		#### Model training
		model_trained=model.train(labeledRDD)
		#broadcast the model
		mb=sc.broadcast(model_trained)
		### Model testing
		Y_pred = model_trained.predict(testRDD2).collect()
		score = accuracy_score(Y_test, Y_pred)
		scores[model.__name__].append(score)
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

