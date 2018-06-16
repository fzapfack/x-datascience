import copy
import numpy as np
class oneVSrest():
    
    def __init__(self, clf):
        self.clf = clf
        
    def fit(self, X_train, y_train):
        clfs = []
        for i in range(10):
            y_new = y_train== i
            clf = copy.copy(self.clf)
            clf.fit(X_train,y_new)
            clfs.append(clf)
        self.clfs = clfs
            
    
    def predict(self, X_test):
        predictions = np.array([clf.decision_function(X_test) for clf in self.clfs])
        self.predictions = predictions
        y_pred = np.argmax(predictions,axis=0)
        return y_pred
    

