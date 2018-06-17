'''
Contains the differents features computed
'''

import pandas as pd
import numpy as np

import nltk
import string
import re

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from gensim.models import word2vec

import networkx as nx

def tokenize(content, remove_stopwords=True, stemming=True):
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
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
    model = model_title(num_features,toload=False)
    vec_avg = np.zeros((node_infoDF.shape[0],num_features),dtype="float32")
    for i in range(vec_avg.shape[0]):
        vec_avg[i] = makeFeatureVec(node_infoDF.title[i], model, num_features)
        if np.sum(np.isnan(vec_avg[i]))>0:
            print i
    return vec_avg

def abstract_avg_word2vec():
    num_features = 300
    model = model_abstract(num_features,toload=False)
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
