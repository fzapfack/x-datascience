import numpy as np
import scipy as sp
import scipy.linalg as linalg

def LDA(X, Y):
    """
	Train a LDA classifier from the training set
	X: training data
	Y: class labels of training data

	"""    

    classLabels = np.unique(Y)
    classNum = len(classLabels)
    datanum, dim = X.shape
    totalMean = np.mean(X,0)

	# partition class labels per label - list of arrays per label
    partition=[np.where(Y==label) for label in classLabels]

	# find mean value per class (per attribute) - list of arrays per label
    classMean = [(np.mean(X[idx],0),len(idx)) for idx in partition]

	# Compute the within-class scatter matrix
    Sw = np.zeros((dim,dim))
	# covariance matrix of each class * fraction of instances for that class 
    for idx in partition:
         Sw = Sw+np.cov(X[idx],rowvar=0)*len(idx)
	# Compute the between-class scatter matrix
    Sb = np.zeros((dim,dim))
    for class_mean, class_size in classMean:
         temp=(class_mean-totalMean)[:,np.newaxis]
         Sb=Sb+class_size*np.dot(temp,np.transpose(temp))


	# Solve the eigenvalue problem for discriminant directions to maximize class separability while simultaneously minimizing
	# the variance within each class
	#eigval, eigvec
    prdct = np.dot(linalg.inv(Sw),Sb)
    eigval, eigvec = linalg.eig(prdct)

    idx = eigval.argsort()[::-1] # Sort eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    W = np.real(eigvec[:,:classNum-1]) # eigenvectors correspond to k-1 largest eigenvalues


	# Project data onto the new LDA space
    X_lda = np.dot(X,W)

	# project the mean vectors of each class onto the LDA space
    projected_centroid = [np.dot(m,W) for m,class_size in classMean]

    return W, projected_centroid, X_lda