from __future__ import division
import numpy as np
from numpy import linalg
from kernels import *

import random
random.seed(42)
import math


class SVM():

    def score(self,X,y):
        y_predict = classif.predict(X)
        correct = np.sum(y_predict == y_test)
        correct=0
        total=len(y_predict)
        for i in range(len(y_predict)):
            if np.array_equal(y_predict[i],y_test[i]):
                correct=correct+1
        print "%d out of %d predictions correct" % (correct, total)

    # alpha = regrularization = 1/n_samples*C
    def __init__(self, kernel=linear_kernel, alpha=0.0000001, n_iter=10, fit_intercept = True):
        self.kernel = kernel
        self.alpha = alpha
        if self.alpha is not None: self.alpha = float(self.alpha)
        self.n_iter = n_iter
        if self.n_iter is not None: self.n_iter = int(self.n_iter)
        if self.n_iter==0:
            print "Doing at least 1 iteration"
            self.n_iter = 1
        self.fit_intercept = fit_intercept
    
    def get_params(self, deep=True):
        return {"alpha": self.alpha, "kernel": self.kernel, "n_iter": self.n_iter, "fit_intercept": self.fit_intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def setattr(self, parameter, value):
        setattr(self, parameter, value)

    def __copy__(self):
        return  SVM().set_params(**self.get_params())

    def fit(self, X, y):
        y=y.astype('float')
        y[y == 0.0] = -1.0 #pourquoi?? -1/1
        
        # fit intercept 
        if self.fit_intercept == True:
            X = np.column_stack((X, np.ones(X.shape[0])))
        n_samples, n_features = X.shape
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
        #initialisation
        alphas_old = np.zeros(n_samples)
        alphas_new = alphas_old.copy()
        for t in range(int(self.n_iter*n_samples)):
            # Pick randomly an observsation
            it = random.randint(0, n_samples-1)
            xit = X[it,:][np.newaxis,:]
            yit = y[it]
            
            # Compute gradient step
            nt = 1/((t+1)*self.alpha)
            
            # Update j!=it (en fait je copie tout)
            alphas_new = alphas_old.copy()
            
            # Update j==it
            # compute yit<wt,phi(xit)>
            sv = alphas_new > 0
            ind_sv = np.arange(len(sv))[sv]
            c=0
            for i_sample in ind_sv:
#                 c = c + yit*K[it,i_sample]*alphas_old[i_sample]
                c = c + y[i_sample]*K[it,i_sample]*alphas_old[i_sample]
            c = yit*nt*c
            if c<1:
                alphas_new[it] = alphas_old[it] + 1
            else :
                alphas_new[it] = alphas_old[it]


            alphas_old = alphas_new
        
        # Strore support vectors
        alphas_new = alphas_new.ravel()
        sv = alphas_new > 0
        self.a = alphas_new[sv]
        self.sv = X[sv,:]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.a), n_samples)      



    def project(self, X):
        if self.fit_intercept == True:
            X = np.column_stack((X, np.ones(X.shape[0])))
        if self.sv is None :
            raise Exception('Model not fitted')
        
        y_predict = np.zeros(X.shape[0])
        for it in range(X.shape[0]):
            c=0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                 c = c + a*sv_y*self.kernel(X[it], sv)
            y_predict[it] = c
        return y_predict
         

    def decision_function(self,X):
        res=self.project(X)
        return res
    
    def predict(self, X):
        res=np.sign(self.project(X))
        return res  

    
