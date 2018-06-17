from numpy import *
import matplotlib.pyplot as plt

# Load the distance matrix
D = loadtxt('distanceMatrix.csv', delimiter=',')
cities = ['Atl','Chi','Den','Hou','LA','Mia','NYC','SF','Sea','WDC']
nCities = D.shape[0] # Get the size of the matrix


k = 2 # e.g. we want to keep 2 dimensions

# MDS algorithm
## TODO: Implement MDS. The new data matrix should have name X
J = eye(nCities) - (1.0/nCities) * ones(nCities)
# Compute matrix B
B = -(1.0/2) * dot(J,dot(pow(D,2),J))
# SVD decomposition of B        
U,L,V = linalg.svd(B)
# Calculate new data
X = dot(U[:,:k],sqrt(diag(L)[:k,:k]))

# Plot distances in two dimensions
plt.figure(1)

# Plot cities in 2D space
plt.subplot(121)
plt.plot(-X[:,0],-X[:,1],'o')
for i in range(len(cities)):
     plt.text(-X[i,0], -X[i,1]+30, cities[i], color='k', ha='center', va='center')

# Plot also a US map
plt.subplot(122)
im = plt.imread("usamap.png")
implot = plt.imshow(im,aspect='auto')
plt.axis('off')
plt.show()


