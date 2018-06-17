# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:14:42 2015

@author: Fabrice ZAPFACK
"""

# ipytho notebook - lance l editeur ipython (voir)

# import library
import pandas as pd

titanic = pd.read_csv('C:\\train.csv')

#import matplot
import matplotlib.pyplot as plt
import numpy as np

titanic.describe()
titanic['Age'].var()
titanic['Embarked'].unique()
titanic[pd.isnull(titanic['Age'])].shape

fig=plt.figure()
ax=fig.add_subplot(111)
ax.hist(titanic['Age'].values,bins=20,range=(titanic['Age'].min(),titanic['Age'].max()))
                
titanic.loc[pd.isnull(titanic['Age']),['Age']]=titanic['Age'].mean()     

titanic.boxplot(column='Age', by='Sex')      
titanic.boxplot(column='Age', by=['Pclass','Sex'])    

def title(name):
    fname=name.split(',')[1]
    ttl=fname.split('.')[0].strip()
    return ttl
    
titanic['Name'].apply(title).unique
titanic['title']=titanic['Name'].apply(title)
titanic.groupby('title').size()

def repl(t):
    if t in ['Mr', 'Mrs','Miss','Master']:
        return t
    else:
        return 'Other'
titanic['title2']=titanic['title'].apply(repl)
titanic.groupby('title2').size()

titanic.boxplot(column='Age', by=['title','Pclass'])   

mean_vals = titanic[['Pclass','Sex','title','Age']].groupby(['Pclass','Sex','title']).agg('mean').reset_index().values
print mean_vals

for r in mean_vals:
    titanic.loc[(pd.isnull(titanic['Age'])) & (titanic['Sex']==r[1]) &
    (titanic['Pclass']==r[0]) & (titanic['title']==r[2]), ['Age']] = r[3]
    
    
gbs=titanic.groupby('Sex')['Survived'].count()

pd.crosstab(titanic['Sex'],titanic['Survived'])
pd.crosstab(titanic['Sex'],titanic['Survived'].astype(bool)).plot(kind='bar')
pd.crosstab(titanic['Sex'],titanic['Survived'].astype(bool)).plot(kind='bar',stacked=True,color=['red','green'])