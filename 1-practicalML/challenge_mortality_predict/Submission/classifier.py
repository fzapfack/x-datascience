from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
import numpy as np



# class Nan(object):

#     def transform(self, X):
#         X2 = X
#         for i in range(X.shape[1]):
#             a = np.nanmax(X[:,i])
#             X2[np.isnan(X[:,i]),i] = a

#         print np.sum(np.isnan(X2))
#         print np.sum(np.isinf(X2))
#         #break
#         return X2

#     def fit(self, X, y=None):
#         return self


def missing(X):
    X2 = X
    for i in range(X.shape[1]):
        a = np.nanmax(X[:,i])
        X2[~np.isfinite(X[:,i]),i] = a
    return X2
class Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = Pipeline([('imputer', Imputer(strategy='most_frequent')),
                             ('rf', RandomForestClassifier(n_estimators=30))])

        X2 = missing(X)
        self.clf.fit(X2, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

