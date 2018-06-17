# -*- coding: utf-8 -*-

import random
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from sklearn import preprocessing
import nltk
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pylab
import sklearn
import scipy
from scipy import stats
import os
import PIL
from PIL import Image
import pickle
import math
from tempfile import TemporaryFile
import re
%matplotlib inline
import random
random.seed(0)
import nltk
import string
import re
nltk.download('stopwords')
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
from gensim.models import word2vec

testing_set = pd.read_csv("testing_set.txt",sep=" ",header=None )
training_set = pd.read_csv("training_set.txt",sep=" ",header=None )
nodeinf = pd.read_csv("node_information.csv", sep=",",header=None )
baseothervartest = pd.read_csv("C:\\Users\\Sylvain\\Desktop\\fab_test.csv",sep=",")
baseothervar= pd.read_csv("C:\\Users\\Sylvain\\Desktop\\fab1.csv",sep=",")
#This code enables to keep only unique elements of a list


def unique(a):
    k=0
    while k < len(a):
        if a[k] in a[k+1:]:
            a.pop(k)
        else:
            k=k+1
    return a
def applytest(x):
    b=0
    for i in range(len(nodeinf)):
        dict1=[]
        dict1=dict1+re.split('\s+', nodeinf[5][i])
        if x in dict1:
            b=b+1
    ret=b/len(nodeinf)
    return(ret)
    
def applyfunc(x):
    x1=re.split('\s+', x)
    g1=[]
    for g in x1:
        g1=g1+[dict0["docfr"][g]]
        
    dictionary= dict(zip(x1, g1))
    return(dictionary)    
    
def applyfunc1(x):
    x1=[]
    x2=[]
    for g2 in x:
        if x[g2]>10**(-4):
            x1=x1+[g2]
            x2=x2+[x[g2]]
    dico = dict(zip(x1,x2))
    return(dico)    
    
    
def applyfunct11(x):
    x1=[]
    x2=[]
    for g2 in x:
        if x[g2]<0.3:
            x1=x1+[g2]
            x2=x2+[x[g2]]
    dico = dict(zip(x1,x2))
    return(dico)    
    
def applyfunc2(x):
    return(len(x))  

def applyfunc3(x):
    di=x[1]
    di1=x[2]
    dc=re.split('\s+', x[0])
    x1=[]
    x2=[]
    for g2 in di:
        c=0
        for j in dc:
            if j==g2:
                c=c+1
        x1=x1+[g2]
        x2=x2+[((1+np.log(1+np.log(c/len(dc))))/(1-b+b*(len(dc)/avglen)))*np.log((N+1)/(int(di1)))]
    dico = dict(zip(x1,x2))
    return(dico)   

    
 def applyfunc5(x):
    x1=[]
    x2=[]
    for g2 in x:
        if x[g2]>1.7:
            x1=x1+[g2]
            x2=x2+[x[g2]]
    dico = dict(zip(x1,x2))
    return(dico)    
    
 def tokenize(content, remove_stopwords=True, stemming=True):
    # remove formatting
    content =  re.sub("\s+", " ", content)
    # convert to lower case
    content = content.lower()
    # remove punctuation (preserving intra-word dashes)
    content = "".join(letter for letter in content if letter not in punct)
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
        tokens = [token for token in tokens if token not in stpwds]
    if stemming==True:
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens   
    
def numcomau(x):
    a1=x["source"]
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,3]
    if pd.isnull(str1)==False:
        b1=re.split(',',str1)
        cst=0
        a2=x["target"]
        str2=nodeinf[nodeinf[0]==a2]
        str2=str2.reset_index(drop=True)
        str2=str2.loc[0,3]   
        if pd.isnull(str2)==False:
            b2=re.split(',', str2)
            for j in range(len(b1)):
                for k in range(len(b2)):
                    if tokenize(b1[j])==tokenize(b2[k]):
                        cst=cst+1
            return(cst)
        else:
             return(0)
    else:
        return(0)

def numcomau1(x):
    a1=x["source"]
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=int(str1.loc[0,1])
    if pd.isnull(str1)==False:
        a2=x["target"]
        str2=nodeinf[nodeinf[0]==a2]
        str2=str2.reset_index(drop=True)
        str2=int(str2.loc[0,1])   
        if pd.isnull(str2)==False:
            return(str1-str2)
        else:
            return(1)
    else:
        return(1)
def numcomau2(x):
    a1=x["source"]
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,2]
    if pd.isnull(str1)==False:
        b1=re.split(' ',str1)
        cst=0
        a2=x["target"]
        str2=nodeinf[nodeinf[0]==a2]
        str2=str2.reset_index(drop=True)
        str2=str2.loc[0,2]   
        if pd.isnull(str2)==False:
            b2=re.split(' ',str2)
            for j in range(len(b1)):
                for k in range(len(b2)):
                    if b1[j]==b2[k]:
                        cst=cst+1
            return(cst)
        else:
            return(0)
    else:
         return(0)


def numcomau3(x):
    a1=x["source"]
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,4]
    if pd.isnull(str1)==False:
        
        cst=0
        a2=x["target"]
        str2=nodeinf[nodeinf[0]==a2]
        str2=str2.reset_index(drop=True)
        str2=str2.loc[0,4]   
        if pd.isnull(str2)==False:

            if str1==str2:
                cst=1
            return(cst)
        else:
            return(0)
    else:
        return(0)


def numcomau4(x):
    a1=x["source"]
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,"tfidf1"]
    if pd.isnull(str1)==False:
        
        cst=0
        a2=x["target"]
        str2=nodeinf[nodeinf[0]==a2]
        str2=str2.reset_index(drop=True)
        str2=str2.loc[0,"tfidf1"]
        if pd.isnull(str2)==False:

            for j in str1:
                for k in str2:
                    if j==k:
                        cst=cst+1
            return(cst)
        else:
             return(0)
    else:
        return(0)


def removepar(mystring):
    if "(" in mystring:
        nb=mystring.count(")")
        for l in range(nb):
            start = mystring.find( '(' )
            end= mystring.find( ')' )
            mystring = mystring[0:start]+mystring[end+1:len(mystring)]
        
        
        if "(" in mystring:
            start = mystring.find( '(' )


            res = mystring[0:start]
            
            return(res)
        else:
            return(mystring)
    else:
        return(mystring)


def fct1(x):
    a1=x
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,3]
    if pd.isnull(str1)==False:
        str1=removepar(str1)
        b1=re.split(', ',str1)
        return(b1)
    else:
        return(str1)


def fct2(x):
    a1=x
    str1=nodeinf[nodeinf[0]==a1]
    str1=str1.reset_index(drop=True)
    str1=str1.loc[0,4]
    return(str1)


def indices(lst, element):
    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)


def fun01(x):
    y=str(x["source"])+","+str(x["target"])
    return(y)
def fun1(x):
    str0=str(int(x["source"]))+","+str(int(x["target"]))
    ind=lis.index(str0)
    return(baseothervar["abstract_dist"][ind])    
def fun2(x):
    str0=str(int(x["source"]))+","+str(int(x["target"]))
    ind=lis.index(str0)
    return(baseothervar["in_degree"][ind])
def fun3(x):
    str0=str(int(x["source"]))+","+str(int(x["target"]))
    ind=lis.index(str0)
    return(baseothervar["out_degree"][ind])
def fun4(x):
    str0=str(int(x["source"]))+","+str(int(x["target"]))
    ind=lis.index(str0)
    return(baseothervar["same_conn_comp"][ind])
def leng(x):
    return(len(x))
def fu(x):
    if pd.isnull(x)==False:
        return(x.split(","))
    else:
        return([])
def fu2(x):
    a=[]
    if x!=[]:
        for g in x:
            if g in diction:
                a=a+[g]
        return(a)
    else:
        return([])
def aut(x):
    return(col[list1.index(x)])


def fu3(x):
    a=[]
    for k in x:
            a=a+[diction[k]]
    return(a)    
def fu4(x):
    return(sum(x))

























    

def mergin_features(training_set,testing_set,baseothervartest,baseothervar):

        
    #We create a dictionary dict0 to keep track of the words in the abstracts.
    dict0=['the']
    for i in range(len(nodeinf)):
        dict1=[]
        dict1=dict1+re.split('\s+', nodeinf[5][i])
        for j in range(len(dict1)):
            if dict1[j] not in dict0:
                dict0=dict0+[dict1[j]]
    dict0=pd.DataFrame(dict0)
    dict0['docfr']=0
    for i in range(len(dict0)):
        b=0
        for j in range(len(nodeinf)):
            dict1=[]
            dict1=dict1+re.split('\s+', nodeinf[5][i])
            if dict0[0][i] in dict1:
                b=b+1
        dict0["docfr"][i]=b/len(nodeinf)
    #'docfr' is the document frequency of each word.
    

    dict0["docfr"]=dict0[0].apply(applytest)
    
    #With the help of the library pickle, we are able to save
    # the variables in a folder and load them easily.
    
    dict0=dict0.set_index([0])

    nodeinf["docfr"]=0
    nodeinf["docfr"]=nodeinf[5].apply(applyfunc)

    nodeinf["docfr"]=nodeinf["docfr"].apply(applyfunc1)

    nodeinf["docfr"]=nodeinf["docfr"].apply(applyfunct11)

    nodeinf["doclen"]=0
    nodeinf["doclen"]=nodeinf["docfr"].apply(applyfunc2)
    N=len(nodeinf)
    avglen=np.mean(nodeinf["doclen"])
    nodeinf["tfidf"]=nodeinf["docfr"]

    
    
    #Thanks to these previous manipulations we are able to compute the tfidf of each word.
    
    
    
    for i in range(len(nodeinf)):
        di=nodeinf["docfr"][i]
        di1=nodeinf["doclen"][i]
        dc=re.split('\s+', nodeinf[5][i])
        x1=[]
        x2=[]
        for g2 in di:
            c=0
            for j in dc:
                if j==g2:
                    c=c+1
            x1=x1+[g2]
            x2=x2+[((1+math.log(1+math.log(c+math.pow(10,-30))))/(1-b+b*(len(dc)/avglen)))*math.log(1/(di[g2]))]
        dico = dict(zip(x1,x2))
        nodeinf["tfidf"][i]=dico
        if i%100==0:
            print(i)
            
    #We remove the words with a small tfidf
    
   
    nodeinf["tfidf1"]=0
    nodeinf["tfidf1"]=nodeinf["tfidf"].apply(applyfunc5)
    training_set.columns=['source','target','label']
    testing_set.columns=['source','target']
    
    
    import nltk
    import string
    import re
    nltk.download('stopwords')
    stpwds = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    punct = string.punctuation.replace("-", "")
    

        
    ##############################################################################
        #Number of common authors
    ##############################################################################
    training_set["num_com_aut"]=0

    training_set["num_com_aut"]=training_set.apply(numcomau,axis=1)
    testing_set["num_com_aut"]=testing_set.apply(numcomau,axis=1)
    
    
    ##############################################################################
    #Temporal difference
    ##############################################################################
    training_set["tempdiff"]=0
    testing_set["tempdiff"]=0

    training_set["tempdiff"]=training_set.apply(numcomau1,axis=1)
    testing_set["tempdiff"]=testing_set.apply(numcomau1,axis=1)
    
    ##############################################################################
    #Number of common words in title
    ##############################################################################
    training_set["numcomwordintitle"]=0
    testing_set["numcomwordintitle"]=0

    training_set["numcomwordintitle"]=training_set.apply(numcomau2,axis=1)
    testing_set["numcomwordintitle"]=testing_set.apply(numcomau2,axis=1)
    
    ##############################################################################
    #Same revue or not
    ##############################################################################
    training_set["samejournal"]=0
    testing_set["samejournal"]=0

    training_set["samejournal"]=training_set.apply(numcomau3,axis=1)
    testing_set["samejournal"]=testing_set.apply(numcomau3,axis=1)
    
    
    ##############################################################################
    #Number of high tfidf words in common in the abstract
    ##############################################################################
    training_set["numofightfidfwordsincom"]=0
    testing_set["numofightfidfwordsincom"]=0

    training_set["numofightfidfwordsincom"]=training_set.apply(numcomau4,axis=1)
    testing_set["numofightfidfwordsincom"]=testing_set.apply(numcomau4,axis=1)
    
    #We train only on the papers which have a positive temporal difference as for
    # the others we know the exact value of the label
    training_set=training_set[training_set["tempdiff"]>0]
    training_set=training_set.reset_index(drop=True)
    training_set["numtimesautcitedpaperinjournal"]=0
    testing_set["numtimesautcitedpaperinjournal"]=0
    
    ##############################################################################
    #How many times did the source author cite the target revue variable
    ##############################################################################
    

    
    dictaut=[]
    for i in range(len(nodeinf)):
        if pd.isnull(nodeinf[3][i])==False:
            str0=removepar(nodeinf[3][i])
            dict1=re.split(', ',str0)
            for j in range(len(dict1)):
                if dict1[j] not in dictaut:
                    dictaut=dictaut+[dict1[j]]
    dictaut=pd.DataFrame(dictaut)
    trset=training_set[training_set["label"]==1].reset_index(drop=True)
    dictaut["papers"]=0

    dictaut["papers"]=dictaut[0].apply(timespapercited)
    trset["autsource"]=0

    trset["autsource"]=trset["source"].apply(fct1)
    trset["journtarget"]=0

    trset["journtarget"]=trset["target"].apply(fct2)
    for i in range(len(trset)):
        a1=trset.loc[i,:]
        b1=a1["autsource"]
        str2=a1["journtarget"]
        if type(b1)==list:
            if pd.isnull(str2)==False:
                for k in b1:
                    str3=dictaut[dictaut[0]==k].reset_index(drop=True).loc[0,"papers"]
                    if str2 in str3:
                        str3[str2]=str3[str2]+1
                    else:
                        str3[str2]=1
    dictaut["asup"]=0
    dictaut["len"]=0
    for i in range(len(dictaut)):
    dictaut.loc[i,"len"]=len(dictaut.loc[i,0])
    dictaut=dictaut.sort(["len"],ascending=False).reset_index(drop=True)
    for i in range(len(dictaut)):
        if dictaut.loc[i,"papers"]=={}:
            dictaut.loc[i,"asup"]=1
    dictaut=dictaut[dictaut["asup"]==0].reset_index(drop=True)
    dictaut=dictaut[dictaut["len"]>3]
    for i in range(len(dictaut)):
        for j in range(i+1,len(dictaut)):
            if dictaut[0][j] in dictaut[0][i]:
                dictaut.loc[i,"asup"]=1
                d1=dictaut.loc[i,"papers"]
                for k in d1:
                    str3=dictaut.loc[j,"papers"]
                    if k in str3:
                        str3[k]=str3[k]+d1[k]
                    else:
                        str3[k]=d1[k]
    dictaut=dictaut[dictaut["asup"]==0].reset_index(drop=True)
    training_set["nbdefoisautcitejournarttarget"]=0
    testing_set["nbdefoisautcitejournarttarget"]=0
    for i in range(len(training_set)):
        a1=training_set["source"][i]
        str1=nodeinf[nodeinf[0]==a1]
        str1=str1.reset_index(drop=True)
        str1=str1.loc[0,3]
        if pd.isnull(str1)==False:
            str1=removepar(str1)
            b1=re.split(',',str1)
            a2=training_set["target"][i]
            str2=nodeinf[nodeinf[0]==a2]
            str2=str2.reset_index(drop=True)
            str2=str2.loc[0,4]   
            if pd.isnull(str2)==False:
                cst=0
                j1=0
                for k in b1:
                    if len(k)>3:
                        if k in list(dictaut[0]):
                            str3=dictaut[dictaut[0]==k].reset_index(drop=True).loc[0,"papers"]
                            if str2 in str3:
                                cst=cst+str3[str2]
                        else:
    
                            for j in range(len(dictaut)):
                                if k in dictaut[0][j]:
                                    str3=dictaut .loc[j,"papers"]
                                    if str2 in str3:
                                        cst=cst+str3[str2]
                                    j1=1+j1
                        
                cst=cst/(len(b1)+j1)
                training_set["nbdefoisautcitejournarttarget"][i]=cst
            else:
                 training_set["nbdefoisautcitejournarttarget"][i]=0
        else:
            training_set["nbdefoisautcitejournarttarget"][i]=0
        if i%100==0:
            print(i/len(training_set)*100)
            
    for i in range(len(testing_set)):
        a1=testing_set["source"][i]
        str1=nodeinf[nodeinf[0]==a1]
        str1=str1.reset_index(drop=True)
        str1=str1.loc[0,3]
        if pd.isnull(str1)==False:
            str1=removepar(str1)
            b1=re.split(',',str1)
            a2=testing_set["target"][i]
            str2=nodeinf[nodeinf[0]==a2]
            str2=str2.reset_index(drop=True)
            str2=str2.loc[0,4]   
            if pd.isnull(str2)==False:
                cst=0
                j1=0
                for k in b1:
                    if len(k)>3:
                        if k in list(dictaut[0]):
                            str3=dictaut[dictaut[0]==k].reset_index(drop=True).loc[0,"papers"]
                            if str2 in str3:
                                cst=cst+str3[str2]
                        else:
    
                            for j in range(len(dictaut)):
                                if k in dictaut[0][j]:
                                    str3=dictaut .loc[j,"papers"]
                                    if str2 in str3:
                                        cst=cst+str3[str2]
                                    j1=1+j1
                        
                cst=cst/(len(b1)+j1)
                testing_set["nbdefoisautcitejournarttarget"][i]=cst
            else:
                 testing_set["nbdefoisautcitejournarttarget"][i]=0
        else:
            testing_set["nbdefoisautcitejournarttarget"][i]=0
        if i%100==0:
            print(i/len(testing_set)*100)
    
    
    
    
    ##############################################################################
    #Depth first search
    ##############################################################################
    adjmatrix = [[0 for x in range(len(nodeinf))] for x in range(len(nodeinf))] 
    list1=list(nodeinf[0])
    for i in range(len(trset)):
        adjmatrix[list1.index(trset["source"][i])][list1.index(trset["target"][i])]=1
        adjmatrix[list1.index(trset["target"][i])][list1.index(trset["source"][i])]=1

    training_set["dfs"]=0
    testing_set["dfs"]=0
    adjmatrix2 = [[0 for x in range(len(nodeinf))] for x in range(len(nodeinf))] 
    dic={}
    for i in range(len(nodeinf)):
        dic[i]=indices(adjmatrix[i], 1)
    
    for i in range(len(nodeinf)):
        for j in range(i+1,len(nodeinf)):
            l1=dic[i]
            l2=dic[j]
            if set(l1).intersection(l2)!=set():
                adjmatrix2[i][j]=1
                adjmatrix2[j][i]=1
    dic2={}
    for i in range(len(nodeinf)):
        dic2[i]=indices(adjmatrix2[i], 1)     
        
    for i in range(len(training_set)):
        l1=list1.index(training_set["source"][i])
        l2=list1.index(training_set["target"][i])
        if l2 in dic2[l1]:
            training_set["dfs"][i]=2
    for i in range(len(testing_set)):
        l1=list1.index(testing_set["source"][i])
        l2=list1.index(testing_set["target"][i])
        if l2 in dic2[l1]:
            testing_set["dfs"][i]=2
    
    for i in range(len(training_set)):
        if training_set["dfs"][i]!=2:
            l1=list1.index(training_set["source"][i])
            l2=list1.index(training_set["target"][i])
            m1=dic[l1]
            m2=dic2[l2]
            if len(set(m1).intersection(m2))>1:
                if l2 in dic2[l1]:
                    training_set["dfs"][i]=3
    for i in range(len(testing_set)):
        if testing_set["dfs"][i]!=2:
            l1=list1.index(testing_set["source"][i])
            l2=list1.index(testing_set["target"][i])
            m1=dic[l1]
            m2=dic2[l2]
            if len(set(m1).intersection(m2))>1:
                if l2 in dic2[l1]:
                    testing_set["dfs"][i]=3
    for i in range(len(training_set)):
        if training_set["dfs"][i]!=2 and training_set["dfs"][i]!=3:
            l1=list1.index(training_set["source"][i])
            l2=list1.index(training_set["target"][i])
            m1=dic2[l1]
            m2=dic2[l2]
            if len(set(m1).intersection(m2))!=0:
                training_set["dfs"][i]=4
            else:
                training_set["dfs"][i]=7
    for i in range(len(testing_set)):
        if testing_set["dfs"][i]!=2 and testing_set["dfs"][i]!=3:
            l1=list1.index(testing_set["source"][i])
            l2=list1.index(testing_set["target"][i])
            m1=dic2[l1]
            m2=dic2[l2]
            if len(set(m1).intersection(m2))!=0:
                testing_set["dfs"][i]=4
            else:
                testing_set["dfs"][i]=7
    
    
    
    
    ##############################################################################
    #Merging with the other vaviables
    ##############################################################################

    training_set["titles_dist"]=0
    training_set["asup"]=0
    testing_set["asup"]=0
    testing_set["titles_dist"]=0
    dico=[]
    dico1=[]
    dico2=[]
    dico3=[]
    dico3=[]
    for i in range(len(baseothervar)):
        dico=dico+[str(baseothervar["source"][i])+","+str(baseothervar["target"][i])]
        if i%10000==0:
            print(i)
        if i==100000:
            dico1=dico
            dico=[]
        if i==200000:
            dico2=dico
            dico=[]
        if i==300000:
            dico3=dico
            dico=[]
    dico=list(set(dico+dico1+dico2+dico3))
    for i in range(len(training_set)):
        if str(training_set["source"][i])+","+str(training_set["target"][i]) not in dico:
            training_set["asup"][i]=1
    baseothervar["ind"]=""

    baseothervar["ind"]=baseothervar.apply(fun01,axis=1)
    lis=list(baseothervar["ind"])
    training_set=training_set[training_set["asup"]==0]
    training_set=training_set.reset_index(drop=True)
    for i in range(len(training_set)):
        str0=str(training_set["source"][i])+","+str(training_set["target"][i])
        ind=lis.index(str0)
        training_set.loc[i,"titles_dist"]=baseothervar["titles_dist"][ind]
    testing_set["titles_dist"]=baseothervartest["titles_dist"]
    training_set["abstract_dist"]=0

    training_set["abstract_dist"]=training_set.apply(fun1,axis=1)
    testing_set["abstract_dist"]=baseothervartest["abstract_dist"]
    training_set["in_degree"]=0
    testing_set["in_degree"]=0

    training_set["in_degree"]=training_set.apply(fun2,axis=1)
    testing_set["in_degree"]=baseothervartest["in_degree"]
    training_set["out_degree"]=0
    testing_set["out_degree"]=0

    training_set["out_degree"]=training_set.apply(fun3,axis=1)
    testing_set["out_degree"]=baseothervartest["out_degree"]
    training_set["same_conn_comp"]=0
    testing_set["same_conn_comp"]=baseothervartest["same_conn_comp"]

    training_set["same_conn_comp"]=training_set.apply(fun4,axis=1)
    training_set["common_neighbors"]=0
    testing_set["common_neighbors"]=0
    for i in range(len(testing_set)):
        if testing_set["deptforsearch"][i]==2:
            l1=list1.index(testing_set["source"][i])
            l2=list1.index(testing_set["target"][i])
            m1=dic[l1]
            m2=dic[l2]
            comcomp=list(set(m2).intersection(m1))
            testing_set["common_neighbors"][i]=len(comcomp)
    for i in range(len(training_set)):
        if training_set["deptforsearch"][i]==2:
            l1=list1.index(training_set["source"][i])
            l2=list1.index(training_set["target"][i])
            m1=dic[l1]
            m2=dic[l2]
            comcomp=list(set(m2).intersection(m1))
            training_set["common_neighbors"][i]=len(comcomp)
    
    
    
    
    ##############################################################################
    #Merging the page rank results
    ##############################################################################
    f = open('pagerankVariables.p', 'rb')
    basepagerank= pickle.load(f)
    f.close()
    basepagerank["len"]=0

    basepagerank["len"]=basepagerank["author"].apply(leng)
    basepagerank=basepagerank[basepagerank["author_authorities"]>10**(-5)]
    list3=list(basepagerank["author"])
    basepagerank=basepagerank.reset_index(drop=True)
    name=list(basepagerank["author"])
    attribute=list(basepagerank["author_authorities"])
    diction=dict(zip(name, attribute))
    nodeinf["aut"]=0

    nodeinf["aut"]=nodeinf[3].apply(fu)
    nodeinf["aut1"]=0

    nodeinf["aut1"]=nodeinf["aut"].apply(fu2)
    col=list(nodeinf["aut1"])
    training_set["authorsource"]=0
    training_set["authortarget"]=0
    testing_set["authorsource"]=0
    testing_set["authortarget"]=0

    training_set["authorsource"]=training_set["source"].apply(aut)
    training_set["authortarget"]=training_set["target"].apply(aut)
    testing_set["authorsource"]=testing_set["source"].apply(aut)
    testing_set["authortarget"]=testing_set["target"].apply(aut)
    liste01=list(training_set["authorsource"])
    liste02=list(training_set["authortarget"])
    liste01=list(testing_set["authorsource"])
    liste02=list(testing_set["authortarget"])

    training_set["authorsourceval"]=0
    testing_set["authorsourceval"]=0
    training_set["authorsourceval"]=training_set["authorsource"].apply(fu3)
    testing_set["authorsourceval"]=testing_set["authorsource"].apply(fu3)
    training_set["authortargetval"]=0
    testing_set["authortargetval"]=0
    training_set["authortargetval"]=training_set["authortarget"].apply(fu3)
    testing_set["authortargetval"]=testing_set["authortarget"].apply(fu3)

    training_set["authorsourceval"]=training_set["authorsourceval"].apply(fu4)
    training_set["authortargetval"]=training_set["authortargetval"].apply(fu4)
    testing_set["authorsourceval"]=testing_set["authorsourceval"].apply(fu4)
    testing_set["authortargetval"]=testing_set["authortargetval"].apply(fu4)
    
    
    
    return(training_set,testing_set)
























