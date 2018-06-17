import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from LDAImp import LDA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data (Wine dataset)
wine = np.genfromtxt('wine_data.csv', delimiter=',')
data = wine[:,1:] # features
labels = wine[:,0] # class labels 

#plot some combinations of dimensions
c=['r','g','b','k']
colors=[c[int(i-1)] for i in labels]
fig = plt.figure(figsize=(15, 8))
for i in range(4):
	ax = fig.add_subplot(241+i, projection='3d')
	ax.scatter(data[:,0+i], data[:,1+i], data[:,2+i],color=colors)
	ax.view_init(4, -72)
	ax = fig.add_subplot(245+i, projection='3d')
	ax.scatter(data[:,9-i], data[:,10-i], data[:,11-i],color=colors)
	ax.view_init(4, -72)
plt.show()

# get LDA projection
W, projected_centroids, X_lda = LDA(data, labels)

#perform PCA to compare with LDA
pca = PCA(n_components=2)
Y = pca.fit_transform(data)

#PLOT them side by side
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(121)
ax.scatter(X_lda[:,0], X_lda[:,1],color=colors)
for ar in projected_centroids:
	ax.scatter(ar[0], ar[1],color='k',s=100)
ax = fig.add_subplot(122)
ax.scatter(Y[:,0], Y[:,1],color=colors)
plt.show()


