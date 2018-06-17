from numpy import *

#data : the data matrix
#k the number of component to return
#return the new data and the variance that was maintained 
def pca(data,k):
	# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
	# Rows of A correspond to observations (wines), columns to variables.
	## TODO: Implement PCA

	# compute the mean
        #mean_data = [mean(data[:,i]) for i in range(size(data,2)) ]
    M = mean(data,0)    
	# subtract the mean (along columns)
    C = data-M
        #substract_data = [data[:,i]-mean_data[i] for i in range(size(data,2)) ]
	# compute covariance matrix
        #cov = dot(transpose(substract_data),substract_data)
    W=dot(transpose(C),C)
	# compute eigenvalues and eigenvectors of covariance matrix
    eigval,eigvec=linalg.eig(W)    
	# Sort eigenvalues (their indexes)
    idx=eigval.argsort()[::-1]
	# Sort eigenvectors according to eigenvalues
    eigvec=eigvec[:,idx]
    eigval=eigval[idx]
	# Project the data to the new space (k-D) and measure how much variance we kept
    eigk=eigvec[:,0:k]    
    data2 = dot(C,eigk)
    perc=sum(eigval[0:k])/sum(eigval)
    
    return (data2,perc)

  
