from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import numpy as np

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
#
# print X.shape
u,s,v = np.linalg.svd(X)
# print u.shape
# print s.shape
# print v.shape
# print s
# print X.shape
#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
# plt.draw()
# plt.show()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#
k=10
s=s[:,np.ne]
X_new = np.dot(u[:,:k],s[:k]).dot(v[:,:k].transpose())

plt.figure(2)
plt.imshow(X_new,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
# plt.draw()
plt.show()


#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#





#=========================================================================



# Plot the optimal rank-k approximation for various values of k)
# # Create a figure with 6 subfigures
# plt.figure(2)

# # Rank 10 approximation
# plt.subplot(321)
# plt.imshow(X10,cmap = cm.Greys_r)
# plt.title('Best rank' + str(5) + ' approximation')
# plt.axis('off')

# # Rank 20 approximation
# plt.subplot(322)
# plt.imshow(X20,cmap = cm.Greys_r)
# plt.title('Best rank' + str(20) + ' approximation')
# plt.axis('off')

# # Rank 50 approximation
# plt.subplot(323)
# plt.imshow(X50,cmap = cm.Greys_r)
# plt.title('Best rank' + str(50) + ' approximation')
# plt.axis('off')

# # Rank 100 approximation
# plt.subplot(324)
# plt.imshow(X100,cmap = cm.Greys_r)
# plt.title('Best rank' + str(100) + ' approximation')
# plt.axis('off')

# # Rank 200 approximation
# plt.subplot(325)
# plt.imshow(X200,cmap = cm.Greys_r)
# plt.title('Best rank' + str(200) + ' approximation')
# plt.axis('off')

# # Original
# plt.subplot(326)
# plt.imshow(X,cmap = cm.Greys_r)
# plt.title('Original image (Rank 480)')
# plt.axis('off')

# plt.draw()


# #================= ADD YOUR CODE HERE ====================================
# # Plot the singular values of the original matrix
# ## TODO: Plot the singular values of X versus their rank k




# #=========================================================================

# plt.show() 

