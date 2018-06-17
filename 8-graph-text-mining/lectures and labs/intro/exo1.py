# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:14:42 2015

@author: Fabrice ZAPFACK
"""

# ipytho notebook - lance l editeur ipython (voir)

# import library
import pandas as pd
df=pd.DataFrame({'A': range(1,5),
                 'B':'foo'})
df

#read csv


museum_list = pd.read_csv('C:\Users\Fabrice ZAPFACK\OneDrive\Documents\school\MasterX\MAP670B\2109\Liste_musees_de_France_utf8.tsv',
                          sep='\t')

museum_list = pd.read_csv('C:\Liste_musees_de_France_utf8.tsv',
                          sep='\t')
                          
museum_list.tail

countDEP = pd.DataFrame(museum_list.groupby(['NOMDEP']).size().reset_index())   
countDEP.colums=['NOMDEP','count']

pop = pd.read_csv('C:\dep_pop_area.csv')

# Upper case
pop['Department2']=pop['Department'].str.upper()

data = pd.merge(countDEP,
                pop,
                left_on='NOMDEP',
                right_on='Department') 
data.ix[data['countDEP'].idxmin()]



#import matplot
import matplotlib.pyplot as plt
import numpy as np

vals=data[['count','Pop_10']].values
                 