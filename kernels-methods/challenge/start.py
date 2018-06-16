import time
import pandas as pd
import numpy as np
from numpy import linalg
from numpy import genfromtxt

from kernels import *
from svm_primal import SVM as svm_primal
# from svm_dual import SVM as svm_dual 
from oneVSrest import *
from preprocess import *




# Change here the paths to data
Ytr = genfromtxt('../Ytr.csv', delimiter=',',skip_header=1,usecols=1)
Xtr = genfromtxt('../Xtr.csv', delimiter=',')
Xte = genfromtxt('../Xte.csv', delimiter=',')
Xtr = transpose_images(Xtr)
Xtr = thresh_images(Xtr)
pca = ImplementedPCA(filt_transform(Xtr),40)
new_xtr = filt_transform(Xtr).dot(pca)
new_xte = filt_transform(Xte).dot(pca)
print("------ End Preprocessing ---------")
print("Training data shape :{}".format(new_xtr.shape))
print("Test data shape :{}".format(new_xte.shape))

start = time.time()
# clf = svm_basile(kernel=polynomial_kernel,C=40)
alpha = 1./float(40*new_xtr.shape[0])
clf = svm_primal(kernel=polynomial_kernel,alpha=alpha, n_iter=50, fit_intercept=True)
classif = oneVSrest(clf)
print("---------Start Fitting (10 models to fit)---------")
classif.fit(new_xtr, Ytr)
y_predict = classif.predict(new_xte)
print("Predicting ...")
print("Fit & predict done in %.3f secs" % (time.time()-start))
d = {'Id': np.arange(1,y_predict.shape[0]+1), 'Prediction': y_predict.astype(int)}
y_out = pd.DataFrame(data=d)

# Output path
y_out.to_csv("out.csv",index=False,sep=',')