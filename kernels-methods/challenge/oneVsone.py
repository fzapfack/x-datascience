import copy
import numpy as np
class oneVSone():
    
    def __init__(self, clf):
        self.clf = clf
        
    def fit(self, X_train, y_train):
        from kernels import *
        seen = []
        clfs = []
        for i1 in range(10):
            for i2 in range(10):
                if i1!=i2 and ((i1,i2) not in seen) and ((i2,i1) not in seen):
                    ind = np.logical_or(y_train==i1,y_train==i2)
                    X_new = X_train[ind,:]
                    X_new = X_train[ind,:]
                    y_new = y_train[ind]
                    y_new[y_new==i2]= -1
                    y_new[y_new==i1]= 1
                    seen.append((i1,i2))
                    clf = copy.copy(self.clf)
                    clf.fit(X_new,y_new)
                    clfs.append(clf)
                    self.seen = seen
                    self.clfs = clfs
    
    def predict(self, X_test):
        predictions = np.array([clf.predict(X_test) for clf in self.clfs])
        self.predictions = predictions
        y_pred = np.zeros(predictions.shape[1])
        for ix in range(predictions.shape[1]):
            votes = []
            for i,p in enumerate(predictions[:,ix]):
                if p==1:
                    votes.append(self.seen[i][0])
                else:
                    votes.append(self.seen[i][1])
            votes = np.array(votes)
            count_votes = np.zeros(10)
            for i in range(10):
                count_votes[i] = np.sum(votes==i)
            y_pred[ix] = np.argmax(count_votes)
        return y_pred
    

