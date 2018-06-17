'''
a=!pwd # ipython magic fontion 
type(str)
str(a)
'''
#exit()
import pandas as pd
from pcaImp import pca
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from functools import partial
#%%

data=pd.read_csv('./cities.txt')
column=list(data.columns)
column.remove('city')
X=data.drop(['city'],axis=1).astype(np.float).values

#we are going to normalize the data so that the unit of measurements will not affect the results
X = normalize(X)

names=data['city'].values

#plot distribution over features
fig, ax1 = plt.subplots(figsize=(14,6))
data.boxplot(column=column)
ax1.xaxis.grid(False)
ax1.set_yscale('log')
plt.xlabel('parameters')
plt.ylabel('values')

###
print "---------------------------"
print names[data['arts'].idxmax()]
plt.show()

#%%
#perform PCA
(Y,perc)=pca(X,2)
print "variance:"+str(perc)
#calculate how important is feature was
scr=np.dot(np.transpose(X),Y)
#scale results to match when we plot them
scr[:,0]= scr[:,0]/(scr.max()- scr.min())
scr[:,1]= scr[:,1]/(scr.max()- scr.min())
 
 
#scatter plot on principal components
##we need this function only to update the scatter plot when we select points
def onpick(event,axes,Y):
	ind = event.ind
	axes.annotate(names[ind], (Y[ind,0],Y[ind,1]))
	plt.draw()
###############
fig, ax1 = plt.subplots(figsize=(14,6))
ax1.scatter(Y[:, 0], Y[:, 1],picker=True)
fig.canvas.mpl_connect('pick_event', partial(onpick, axes=ax1, Y=Y))
for i,v in enumerate(column):
	ax1.plot([0, scr[i,0]], [0, scr[i,1]], 'r-',linewidth=2,)
	plt.text(scr[i,0]* 1.15, scr[i,1] * 1.15, v, color='r', ha='center', va='center')


ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')

ax1.xaxis.grid(True)
ax1.yaxis.grid(True)
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')




plt.show()

