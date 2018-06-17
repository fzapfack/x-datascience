#1st file

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn import svm
import time
import string
import math
from sklearn.metrics import roc_curve, auc
from MyGraph import createGraphFeatures

#function needed to convert distances to probabilities

def predict_proba(X,model):
	f = np.vectorize(lambda x: 1/(1+np.exp(-x)))
	raw_predictions = model.decision_function(X)
	platt_predictions = f(raw_predictions)
	probs = platt_predictions / platt_predictions.sum(axis=1)[:, None]
	return probs

#parameters for graph of words
sliding_window = 2
#a dictionary with all  the idfs (needed for the tw-idf)
idfs = {}

# the labelled training data
cols = ['class', 'text']
train = pd.read_csv("./data/r8-train-stemmed.txt", sep="\t", header=None, names=cols)

print train.shape

# num of docs= rows
num_documents = train.shape[0]

#%%
clean_train_documents=train['text'].values

#use pandas 'magic' to get a list of words from all documents and then get the set of unique words
unique_words=list(set(train['text'].str.split(' ').apply(pd.Series,1).stack().values))
print "Unique words:"+str(len(unique_words))

print "Building features..."
#tf-idf features on train data
start = time.time()
'''fit_transform() does two functions: First, it fits the model
and learns the vocabulary; second, it transforms our training data
into feature vectors. The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_documents)'''
tfidf_vect = TfidfVectorizer(analyzer = "word",lowercase= True,norm=None )
#features = tfidf_vect.fit_transform(clean_train_documents)

#tw-idf features on train data
features, idfs_learned, nodes= createGraphFeatures(num_documents,clean_train_documents,unique_words,sliding_window,True,idfs)
end = time.time()
print "Total time to build features:\t"+str(end - start)


#%%
print "Training the classifier..."


start = time.time()


clf = svm.LinearSVC(loss="hinge")
Y = train['class']
#build a dictionary to assign numerical values to class labels
class_to_num={}
classLabels=train['class'].unique()

for i,val in enumerate(classLabels):
	class_to_num[val]=i
print "Class correspondence"	
print class_to_num
#map the values of the class to the numbers
y=train['class'].map(class_to_num).values

print "Number of classes:"+str(len(classLabels))
model = clf.fit( features, y )
end = time.time()
print "Total time to train classifier:\t"+str(end - start)

#%%
##################

## Testing set
test = pd.read_csv("./data/r8-test-stemmed.txt", sep="\t", header=None, names=cols)

print test.shape

# Get the number of documents based on the data frame column size
num_test_documents = test.shape[0]
clean_test_documents = test['text'].values

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool. 
print "Creating features for test set"

# Careful: here we use transform and not fit_transform
#features_test = tfidf_vect.transform(clean_test_documents)
#similar to tw-idf
features_test,idfs,nodes = createGraphFeatures(num_test_documents,clean_test_documents,nodes,sliding_window,False,idfs_learned)


#map the values of the class to the numbers
y_test=test['class'].map(class_to_num).values
lb = preprocessing.LabelBinarizer()
lb.fit(np.arange(len(classLabels)))
y_test_vector=lb.transform(y_test)
pred_test = model.predict(features_test)

met = metrics.classification_report(y_test, pred_test, target_names=classLabels, digits=4)
print met

probs = predict_proba(features_test,model) 
	
precision, recall, threshold = metrics.precision_recall_curve(y_test_vector.ravel(),probs.ravel())

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="lower left")
plt.show()



