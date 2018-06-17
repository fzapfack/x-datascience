from numpy import *

#D :distance matrix
#k: number of vectors to use
def mds(D,k):
    nelem = D.shape[0]
    J = eye(nelem) - (1.0/nelem) * ones(nelem)
    # Compute matrix B
    B = -(1.0/2) * dot(J,dot(pow(D,2),J))
    # SVD decomposition of B        
    U,L,V = linalg.svd(B)
    # Calculate new data
    X = dot(U[:,:k],sqrt(diag(L)[:k,:k]))
    return X #return product of Uk with the square root of Sk
	# MDS algorithm
	## Implement MDS. 
	#double centering
	# Compute matrix B
	# SVD decomposition of B  (eigen values are sorted)
    