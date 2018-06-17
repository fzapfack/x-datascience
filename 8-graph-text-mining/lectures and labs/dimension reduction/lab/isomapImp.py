import sklearn.neighbors as nb
import sklearn.utils.graph as ug
from numpy import *

def isomap(D,k,n_neighbors):
	#k nearest neighbour algorithm
    knbr = nb.NearestNeighbors(n_neighbors=n_neighbors)
    knbr.fit(D)
	#neighbour graph where the edges are weighted with the euclidean distance
    #kng=nb.kneighbors_graph(knbr,n_neighbors,mode='distance')
    kng=knbr.kneighbors_graph(D,n_neighbors,mode='distance')
	
	#graph distances 
    Dist_g=ug.graph_shortest_path(kng,directed=False, method='auto')
	#Dist_g= 
	#the rest is just like mds
    nelem = Dist_g.shape[0]
    J = eye(nelem) - (1.0/nelem) * ones(nelem)
	# Compute matrix B
    B = -(1.0/2) * dot(J,dot(pow(Dist_g,2),J))
	# SVD decomposition of B  
    U,L,V = linalg.svd(B)
    return dot(U[:,:k],sqrt(diag(L)[:k,:k]))