
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


trainDF = pd.read_csv('./data/training_input.csv')
testDF = pd.read_csv('./data/testing_input.csv')
otherDF = pd.read_csv('./data/challenge_output_data_training_file_prediction_of'
                      '_transaction_volumes_in_financial_markets.csv', sep=';')

def interpol(trainDF,otherDF,missing_ratio):
    ## Rename columns 
    trainDF2 = trainDF.copy()
    cols = list(trainDF2.columns)
    cols_new = list(trainDF2.columns)
    cols_new[3:]=range(trainDF2.shape[1]-3)
    trainDF2=trainDF2.rename(columns=dict(zip(cols, cols_new)))
    
    ## Remove negative values
    trainDF2[trainDF2<0]=np.nan
    
    ## remove line with too much missing values
    trainDF3=trainDF2.drop(['date','product_id','ID'],axis=1)
    ratio_missing=trainDF3.isnull().sum(axis=1).div(trainDF3.shape[1])
    missing_index=ratio_missing[ratio_missing>missing_ratio].index
    trainDF_drop = trainDF2.drop(missing_index,axis=0)
    
    ## Linear interpolation
    interpolatedDF=trainDF_drop.drop(['date','product_id','ID'],
                                 axis=1).interpolate(method='linear', limit_direction='both',limit=54, axis=1)#54*0.1
    
    ## Add Target
    interpolatedDF2 = pd.concat([trainDF_drop[['ID','product_id']],interpolatedDF],axis=1)# faut il ajouter la date et product id                           
    interpolatedDF2 = pd.merge(interpolatedDF2,otherDF,how='left', on=['ID'])
    
    ## Interpolate the dropped lines
    trainDF_missing = trainDF2.iloc[missing_index,:].reset_index(drop=True)
    ## Compute the mean by date
    trainDF_mean_date = trainDF2.groupby(by=["date"]).median() # NaN => interpolation
    trainDF_mean_date = trainDF_mean_date.drop(['product_id','ID'],
                                 axis=1).interpolate(method='linear', limit_direction='both',limit=50, axis=1)
    ## Cpmpute the mean by product
    trainDF_mean_product = trainDF2.groupby(by="product_id").median() # no NaN
    trainDF_mean_product = trainDF_mean_product.drop(['date','ID'],axis=1)
    
    ## Do a linear interpolation
    trainDF_missing_interp = trainDF_missing.drop(['date','product_id','ID'],
                                 axis=1).interpolate(method='linear', limit_direction='both',limit=54, axis=1)
    
    ## Fill missing values by combinin linear interpolation mean by date and mean by product
    trainDF_missing2 = trainDF_missing.copy()
    for i in range(trainDF_missing2.shape[0]):
    # for i in range(10):
        product_id = trainDF_missing2["product_id"][i]
        mean_products = trainDF_mean_product[trainDF_mean_product.index==product_id].iloc[0,:]
        a = trainDF_missing2.iloc[i,3:]
        index = a.isnull()

        date = trainDF_missing2["date"][i]
        mean_date = trainDF_mean_date[trainDF_mean_date.index==date].iloc[0,:]

        lin_interp = trainDF_missing_interp.iloc[i,:]
        if a.isnull().sum()<=52:
            coefs = [1./np.abs(lin_interp.mean()-a.mean()),1./np.abs(mean_date.mean()-a.mean()),1./np.abs(mean_products.mean()-a.mean())]
            coefs = coefs/np.sum(coefs)
            a[index] = coefs[0]*lin_interp[index]+ coefs[1]*mean_date[index]+ coefs[2]*mean_products[index]
        else:
            a = 0.5*mean_date + 0.5*mean_products
        trainDF_missing2.iloc[i,3:] = a
        if trainDF_missing2.iloc[i,3:].isnull().sum()!=0:
            print i
            
    ## Add Target
    trainDF_missing2 = pd.merge(trainDF_missing2,otherDF,how='left', on=['ID'])
    
    ## Concatenate both interpolations
    trainingDF = interpolatedDF2.append(trainDF_missing2.drop(['date'],axis=1))
    
    return trainingDF

# MAPE
from sklearn.utils import check_array
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

for missing_ratio in np.linspace(0.1,0.6,5):
    trainingDF=interpol(trainDF,otherDF,missing_ratio)
    print trainingDF.isnull().sum().sum()
    print trainingDF.shape

    features = trainingDF.drop(['ID','TARGET'], axis=1)
    X_columns = trainingDF.columns.drop(['ID','TARGET'])
    X = features.values
    y = trainingDF['TARGET'].values

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # reg = LinearRegression()
    reg = RandomForestRegressor(n_estimators=10,random_state=40)
    reg.fit(X_train, (y_train-np.mean(y_train))/np.std(y_train))
    y_predict = reg.predict(X_test)
    score = mean_absolute_percentage_error(y_test, (y_predict*np.std(y_train))+np.mean(y_train) )
    print score
