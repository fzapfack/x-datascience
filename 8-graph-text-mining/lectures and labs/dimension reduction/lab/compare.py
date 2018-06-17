from time import time
from numpy import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from scipy.spatial import distance as spd
import pcaImp
import mdsImp
import isomapImp


# The next line is required for plotting only
Axes3D

n_points = 2000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
#X, color = datasets.samples_generator.make_swiss_roll(n_points, random_state=0)
n_components=2
n_neighbors=5

fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

#------PCA--------our implementation
t0 = time()
(Y,perc)=pcaImp.pca(X,n_components)
t1 = time()
print("PCA(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(252)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("PCA(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------

#------MDS--------our implementation (classical MDS)
t0 = time()
D=spd.squareform(spd.pdist(X,'euclidean'))
Y=mdsImp.mds(D,n_components)
t1 = time()
print("MDS(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(253)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------
#------isomap--------our implementation (with MDS)
t0 = time()

'''
import sklearn.neighbors as nb
import sklearn.utils.graph as ug
knbr = nb.NearestNeighbors(n_neighbors=n_neighbors)
knbr.fit(X)
kng=nb.kneighbors_graph(knbr,n_neighbors,mode='distance')
kng=knbr.kneighbors_graph(X,n_neighbors=n_neighbors,mode='distance')

X = [[0], [3], [1]]
neigh = nb.NearestNeighbors(n_neighbors=n_neighbors)
neigh.fit(X)
kng=neigh.kneighbors_graph(X=X,n_neighbors=n_neighbors,mode='distance')
'''
Y=isomapImp.isomap(X,n_components,n_neighbors)
t1 = time()
print("Isomap(imp): %.2g sec" % (t1 - t0))
ax = fig.add_subplot(254)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap(imp) (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#-----------------
''' 
#----PCA---------- sklearn implementation 

t0 = time()
pca = PCA(n_components=n_components)
Y = pca.fit_transform(X)
t1 = time()
print("PCA: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(257)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("PCA (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#--------------------
#----MDS---------- sklearn implementation (Stress minimization-majorization algorithm SMACOF)
t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Y = mds.fit_transform(X)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#---------------
#----Isomap---------- sklearn implementation (with kernel PCA)
t0 = time()
Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
t1 = time()
print("Isomap: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
plt.title("Isomap (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
#--------------------

'''

plt.show()