from __future__ import division
 
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor, AdaBoostRegressor,VotingClassifier,GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso,TheilSenRegressor, LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR
from sklearn import neighbors
from scipy.sparse import csc_matrix,csr_matrix
import numpy as np
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
 
class Regressor(BaseEstimator):
    def __init__(self):
 
        self.regressorName="gb"
        if self.regressorName=="rf":
            self.clf= RandomForestRegressor(n_estimators=400, max_depth=63,max_features=50, n_jobs=-1)
        elif self.regressorName=="gb":
 
            self.clf= GradientBoostingRegressor(alpha=0.9, init=None,max_depth=3, learning_rate=0.2, loss='ls'
                                ,max_features=None,min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0
                                ,n_estimators=2500,presort='auto', random_state=None, subsample=1.0, verbose=0,warm_start=True)
            #self.clf =GridSearchCV(estimator=gb, param_grid=self.getParamGrid(),scoring='mean_squared_error',cv=3,n_jobs=-1)
            #self.clf=gb
        elif self.regressorName=="ridge":
            self.clf = RidgeCV(alphas=(0.01, 0.1), fit_intercept=True, normalize=False, scoring=None, cv=5, gcv_mode=None, store_cv_values=False)
        elif self.regressorName=="linear":
            self.clf = LinearRegression(alpha=0.01,max_iter=5000)
        elif self.regressorName=="lasso":
            self.clf = LassoCV(cv=10)
        elif self.regressorName=="svr":
             self.clf = SVR(kernel='rbf',C=0.2, gamma=0.01)
        elif self.regressorName=="knn":
            self.clf = neighbors.KNeighborsRegressor(1, weights='distance',n_jobs=-1)
        elif self.regressorName=="gauss":
            self.clf = TheilSenRegressor()
 
    def fit(self, X, y):
        #X=csc_matrix(X)
        self.clf.fit(X, y)
        #print self.clf.best_estimator_
 
    def predict(self, X):
        #X=csr_matrix(X)
        return self.clf.predict(X)
 
    def getRegressor(self):
        return self.clf
 
    def getRegressorName(self):
        return self.regressorName
 
    def getParamGrid(self):
        if self.regressorName=="rf":
            defaultGrid=[None]
            maxDepthGrid=np.arange(10,70,7)
            maxFeaturesGrid=["sqrt","log2",None]
            maxTreesGrid=np.arange(10,100,10)
            param_grid = {'max_features': defaultGrid}
        elif self.regressorName == "gb":
            #maxDepthGrid=np.arange(3,20,5)
            learningRateGrid=np.arange(50,100,10)
            #param_grid = {'max_depth': maxDepthGrid}
            #param_grid={'loss':['ls', 'lad', 'huber', 'quantile']}
            param_grid={'alpha':[0.9]}
        return param_grid