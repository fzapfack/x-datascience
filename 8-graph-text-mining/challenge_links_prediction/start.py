import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
random.seed(0)

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing

# from preprocess import *

########### Features to be computed ###############
import nltk
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
import string
import re

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from gensim.models import word2vec

import networkx as nx

def tokenize(content, remove_stopwords=True, stemming=True):
    
    
    stemmer = nltk.stem.PorterStemmer()
    punct = string.punctuation.replace("-", "")
    # remove formatting
    content =  re.sub("\s+", " ", content)
    # convert to lower case
    content = content.lower()
    # remove punctuation (preserving intra-word dashes)
    content = "".join(letter for letter in content if letter not in punct)
    content = re.sub("[^a-zA-Z]"," ", content)
    # remove dashes attached to words but that are not intra-word
    content = re.sub("[^[:alnum:]['-]", " ", content)
    content = re.sub("[^[:alnum:][-']", " ", content)
    # remove extra white space
    content = re.sub(" +"," ", content)
    # remove leading and trailing white space
    content = content.strip()
    # tokenize
    tokens = content.split(" ")
    # remove stopwords
    if remove_stopwords==True:
        tokens = [token for token in tokens if token not in stpwds and len(token)>2]
    if stemming==True:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def overlap(txt1, txt2):
    a = tokenize(txt1)
    b = tokenize(txt2)
    o = len(set(a).intersection(set(b)))
    return o

def common_auths(id1,id2):
    source_auth = node_infoDF.ix[id1,"authors"]
    target_auth = node_infoDF.ix[id2,"authors"]
    if type(source_auth)==float or type(target_auth)==float :
#         if np.isnan(source_auth) or np.isnan(target_auth): #a simplifier
        return 0
    else:
        return len(set(source_auth.split(",")).intersection(set(target_auth.split(","))))

def temp_difference(id1,id2):
    source_year = node_infoDF.ix[id1,"year"]
    target_year = node_infoDF.ix[id2,"year"]
    return source_year-target_year

def model_abstract(num_features=300,toload=False):
    # tester si on train sur un autre corpus plus grand avant
    if toload==True:
        model = word2vec.Word2Vec.load("word2vec_abstract")
    else:
        sentences = []
        for i in range(node_infoDF.shape[0]):
            sentences.append(tokenize(node_infoDF.abstract[i],remove_stopwords=False, stemming=False))
        # Set values for various parameters
        sg=0 # 0 CBOW, 1 Skip-gram
        num_features = num_features    # Word vector dimensionality                      
        min_word_count = 10   # Minimum word count                        
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size                                                                                    
        downsampling = 1e-3   # Downsample setting for frequent words to e-5

        # Initialize and train the model (this will take some time)
        
        print "Training model..."
        model = word2vec.Word2Vec(sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)

        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "word2vec_abstract"
        model.save(model_name)
    return model

def model_title(num_features=300, toload=False):
    if toload==True:
        model = word2vec.Word2Vec.load("word2vec_title")
    else:
        sentences = []
        for i in range(node_infoDF.shape[0]):
            sentences.append(tokenize(node_infoDF.title[i],remove_stopwords=False, stemming=False))
        # Set values for various parameters
        sg=0 # 0 CBOW, 1 Skip-gram
        num_features = num_features    # Word vector dimensionality                      
        min_word_count = 1   # Minimum word count                        
        num_workers = 4       # Number of threads to run in parallel
        context = 10          # Context window size                                                                                    
        downsampling = 1e-3   # Downsample setting for frequent words to e-5

        # Initialize and train the model (this will take some time)
        print "Training model..."
        model = word2vec.Word2Vec(sentences, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = context, sample = downsampling)

        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and 
        # save the model for later use. You can load it later using Word2Vec.load()
        model_name = "word2vec_title"
        model.save(model_name)
    return model

def makeFeatureVec(doc, model, num_features):
    words = tokenize(doc,remove_stopwords=True, stemming=False)
    featureVec = np.zeros((num_features,),dtype="float32")
    index2word_set = set(model.index2word)
    nwords = 0
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    if nwords!=0:
        featureVec = np.divide(featureVec,nwords)
    return featureVec

def title_avg_word2vec():
    num_features = 300
    model = model_title(num_features,toload=True)
    print model
    vec_avg = np.zeros((node_infoDF.shape[0],num_features),dtype="float32")
    for i in range(vec_avg.shape[0]):
        vec_avg[i] = makeFeatureVec(node_infoDF.title[i], model, num_features)
        if np.sum(np.isnan(vec_avg[i]))>0:
            print i
    return vec_avg

def abstract_avg_word2vec():
    num_features = 300
    model = model_abstract(num_features,toload=True)
    vec_avg = np.zeros((node_infoDF.shape[0],num_features),dtype="float32")
    for i in range(vec_avg.shape[0]):
        vec_avg[i] = makeFeatureVec(node_infoDF.abstract[i], model, num_features)
        if np.sum(np.isnan(vec_avg[i]))>0:
            print i
    return vec_avg

from sklearn.metrics.pairwise import cosine_similarity as cosine
def cos_similarity(matrix, i1, i2):
    sim = cosine(matrix[i1,:][np.newaxis,:],matrix[i2,:][np.newaxis,:])  
    return (round(sim, 4))

def create_graph():
    G = nx.DiGraph()
    for i in range(trainDF.shape[0]):
        if trainDF.label[i]==1:
            source_ID = trainDF.source[i]
            target_ID = trainDF.target[i]
            G.add_edges_from([(source_ID,target_ID)])
    return G

def same_comp(G,n1,n2):
    a = nx.node_connected_component(G,n1)
    return int(n2 in a)

def neighbors(G,n1,n2):
    a = G.neighbors(n1)
    b = G.neighbors(n2)
    return len(set(a).intersection(set(b)))

################ End of features ############
#compute features from data set
def compute_features(trainDF):
    for i in xrange(trainDF.shape[0]):
	    source_ID = trainDF.source[i]
	    target_ID = trainDF.target[i]
	    
	    # Indices
	    a = np.arange(node_infoDF.shape[0])
	    i1 = a[(node_infoDF.ID==source_ID).values][0]
	    i2 = a[(node_infoDF.ID==target_ID).values][0]
	    # prof features
	#     source_title = node_infoDF.ix[node_infoDF.ID==source_ID,"title"].values[0]
	#     target_title = node_infoDF.ix[node_infoDF.ID==target_ID,"title"].values[0]
	    source_title = node_infoDF.ix[i1,"title"]
	    target_title = node_infoDF.ix[i2,"title"]
	    trainDF.ix[i,'overlap_title'] = overlap(source_title, source_title)
	    trainDF.ix[i,'comm_auth'] = common_auths(i1, i2)
	    trainDF.ix[i,'temp_diff'] = temp_difference(i1, i2)
	    
	    # word2vec
	    trainDF.ix[i,'titles_dist'] = cos_similarity(titles_avg, i1, i2)
	    trainDF.ix[i,'abstract_dist'] = cos_similarity(abstracts_avg, i1, i2)
	    
	    #Graph
	    if source_ID not in nodes or target_ID not in nodes:
	        trainDF.ix[i,'same_conn_comp'] = 0
	        trainDF.ix[i,'in_degree'] = 0
	        trainDF.ix[i,'out_degree'] = 0
	        # trainDF['common_neighbors'] = 0
	        if source_ID in nodes:
	            trainDF.ix[i,'out_degree'] = G.out_degree(source_ID)
	        if target_ID in nodes:
	            trainDF.ix[i,'in_degree'] = G.in_degree(target_ID)
	    else:
	        trainDF.ix[i,'in_degree'] = G.in_degree(target_ID)
	        trainDF.ix[i,'out_degree'] = G.out_degree(source_ID)
	        trainDF.ix[i,'same_conn_comp'] = same_comp(G2,source_ID,target_ID)
	    	# trainDF['common_neighbors'] = neighbors(G,source_ID,target_ID)
	    if i % 10000 == 0:
	        print (i, "training examples processsed")
    return trainDF

# function used to create a smaller data set
def small_dataset(big,size=5e4,balanced=True):
    if balanced==True:
        pos = big[big.label==1]
        to_keep = random.sample(range(pos.shape[0]), k=int(round(size/2)))
        small1 = pos.iloc[to_keep]
        neg = big[big.label==0]
        to_keep = random.sample(range(neg.shape[0]), k=int(round(size/2)))
        small2 = neg.iloc[to_keep]
        df = pd.concat([small1, small2], axis=0)
        df = df.reindex(np.random.permutation(df.index))
        df = df.reset_index(drop=True)
    else:
        to_keep = random.sample(range(big.shape[0]), k=int(round(size)))
        df = big.ix[to_keep,:].reset_index(drop=True)
    return df


## Read data (in pandas dataFrame) => Please change here path to files

first_train_DF = pd.read_csv("../Data/training_set.txt",header=None,sep=" ")
first_train_DF.columns=['source','target','label']
testDF = pd.read_csv("../Data/testing_set.txt",header=None,sep=" ")
testDF.columns=['source','target']
node_infoDF = pd.read_csv("../Data/node_information.csv",header=None)
node_infoDF.columns=['ID','year','title','authors','journal','abstract']
print("shapes : train={}, test={}, nodes={}".format(first_train_DF.shape, testDF.shape, node_infoDF.shape))

#We will compute the features only on a subset of the given information (only half of the given size will be used for training)
# Please change the size for a good tradeoff time/classification performance
size = 1e5
trainDF = small_dataset(first_train_DF,size=size,balanced=True)
print("using only a subset of data (balanced or not), subset shape:",trainDF.shape)


## Feature extraction
trainDF['overlap_title'] = None
trainDF['temp_diff'] = None
trainDF['comm_auth'] = None
trainDF['titles_dist'] = None
trainDF['abstract_dist'] = None
trainDF['in_degree'] = None
trainDF['out_degree'] = None
trainDF['same_conn_comp'] = None
# trainDF['common_neighbors'] = None

testDF['overlap_title'] = None
testDF['temp_diff'] = None
testDF['comm_auth'] = None
testDF['titles_dist'] = None
testDF['abstract_dist'] = None
testDF['in_degree'] = None
testDF['out_degree'] = None
testDF['same_conn_comp'] = None
# testDF['common_neighbors'] = None

titles_avg = title_avg_word2vec()
abstracts_avg = abstract_avg_word2vec()
G = create_graph()
nodes = G.nodes()
G2 = nx.Graph(G)
# trainDF = trainDF.apply(compute_features,axis=1)
# testDF = testDF.apply(compute_features,axis=1)
trainDF = compute_features(trainDF)
testDF = compute_features(testDF)
print("End features computation, new shape:")
print(trainDF.shape)


# we will create an independant validation set (in other to compute the coefficients on the 3 models votes)
trainDF2 = trainDF.iloc[:trainDF.shape[0]/2]
validDF = small_dataset(trainDF.iloc[trainDF.shape[0]/2:],size=size/10,balanced=True)
a = trainDF2.ix[:,3:]
features = a.columns
X_train = a.values
y_train = trainDF2["label"].values

a = validDF.ix[:,3:]
X_valid = a.values
y_valid = validDF["label"].values

a = testDF.ix[:,2:]
X_test = a.values

# # scale
# X = preprocessing.scale(X)
print (X_train.shape, y_train.shape)
print (X_valid.shape,y_valid.shape)
print (X_test.shape)
models = [svm.LinearSVC(C=0.1, fit_intercept=False),
          RandomForestClassifier(random_state=40,bootstrap=False, min_samples_leaf=1,
            n_estimators=100,min_samples_split=5,criterion='gini',max_features=5,max_depth=15),
          GradientBoostingClassifier(max_depth=5,loss="exponential",n_estimators=200,random_state=42,verbose=0),
          xgb.XGBClassifier(max_depth=6)]
y_pred_valid = np.zeros([X_valid.shape[0],len(models)])
for i in range(len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    print ("Model fitted")
    y_pred_valid[:,i] = model.predict(X_valid)
    # y_pred_test[:,i] = model.predict(X_test)
print ("Models scores validation dataset:")
print [metrics.f1_score(y_valid,y_pred_valid[:,i]) for i in range(y_pred_valid.shape[1])]

## Moyenne ponderee
coefs = np.array( [metrics.f1_score(y_valid,y_pred_valid[:,i]) for i in range(y_pred_valid.shape[1])])
coefs = np.exp(100*coefs)
coefs = coefs/np.sum(coefs)
print("coefficients obtainend from validation dataset")
print(coefs)
# coefs = np.array([0.22,0.33,0.44])

y_pred = np.zeros(X_test.shape[0])
y_pred_valid = np.zeros(X_valid.shape[0])
for i in range(len(models)):
    model = models[i]
    y_pred = y_pred + coefs[i]*model.predict(X_test)
    y_pred_valid = y_pred_valid + coefs[i]*model.predict(X_valid)
y_pred = np.round(y_pred)    
a=(testDF.temp_diff<0).values
y_pred[a]=0

y_pred_valid = np.round(y_pred_valid)    
print ("score validation dataset : ",metrics.f1_score(y_valid,y_pred_valid) )


y_out = pd.DataFrame()
y_out["id"]=np.arange(0,y_pred.shape[0])
y_out["category"] = y_pred.astype(int)
y_out.to_csv("final.csv",index=False,sep=',')
