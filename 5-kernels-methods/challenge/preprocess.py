import time
from scipy import signal
import pandas as pd
import numpy as np


############ Denoising with Walvet######################################
from scipy.ndimage.filters import convolve
def convolution(image_in,step):
    C1=1./16
    C2=1./4
    C3=3./8
    KSize=4*step+1
    KS2=KSize/2
    Kernel=np.zeros((KSize,1))
    Kernel[0]=C1
    Kernel[KSize-1]=C1
    Kernel[KS2+step]=C2
    Kernel[KS2-step]=C2
    Kernel[KS2]=C3
    
    z = convolve(image_in,Kernel)
    kernelY = np.transpose(Kernel)
    Im_out = convolve(z, kernelY)
    return Im_out

def atrou(image, scale):
    WT=-1
    Vsize=image.shape
    Nx=Vsize[0]
    Ny=Vsize[1]
    Nz = scale
    NStep=scale-1
    WT = np.zeros((Nx,Ny,Nz))
    Im_in=image
    Im_out=np.zeros((Nx,Ny))
    Step_trou=1
    for i in range(NStep):
        Im_aux=convolution(Im_in,Step_trou)
        WT[:,:,i]=Im_in-Im_aux
        Im_in=Im_aux
    WT[:,:,NStep]=Im_aux
    return WT

def backward(WT):
    RecIma=0
    vs=WT.shape
    Nscale = vs[2]
    NStep=Nscale-1
    for j in range(Nscale) :
        RecIma = RecIma + WT[:,:,j]
    return RecIma

def Soft(data,treshold):
    local=data.copy()
    local[(local>=-treshold)&(local<treshold)]=0
    local[local<-treshold]=local[local<-treshold]+treshold
    local[local>treshold]=local[local>treshold]-treshold
    return local

from numpy import mean, absolute,median
def SoftThrd(image):
    WT=atrou(image,5)
    sol=WT.copy()
    Nz=WT.shape[2]
    rho=1.4826
    for i in range(Nz-1):
        sigma=rho*median(absolute(WT[:,:,i] - median(WT[:,:,i])))
        t=4*sigma
        sol[:,:,i]=Soft(WT[:,:,i],t)
    sol[:,:,Nz-1]=WT[:,:,Nz-1]
    FinalSol=backward(sol)
    return FinalSol


def HardThrd(image):
    WT=atrou(image,5)
    sol=WT.copy()
    Nz=WT.shape[2]
    rho=1.4826
    for i in range(Nz-1):
        sigma=rho*median(absolute(WT[:,:,i] - median(WT[:,:,i])))
        t=3*sigma
        sol[:,:,i]=Hard(WT[:,:,i],t)
    sol[:,:,Nz-1]=WT[:,:,Nz-1]
    FinalSol=backward(sol)
    return FinalSol

################################## END DENOISING #############################

################################# PCA  ######################################

def ImplementedPCA(X,n_component):
    n_samples,n_features=X.shape
    cov_mat = np.cov(X.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]
    eig_pairs=sorted(eig_pairs, key=lambda tup: tup[0])
    eig_pairs.reverse()
    matrix_w=np.zeros((n_features,n_component))
    for k in range(n_component):
        matrix_w[:,k]= eig_pairs[k][1]
    return matrix_w

################################# edge detector  ######################################

def filt_transform(X):
    X_filt = X.copy()
    for i in range(X.shape[0]):
        filt = np.array([[ 0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])

        filt = filt/5.
        X_filt[i,:] = signal.convolve2d(X[i,:].reshape([28,28]), filt, boundary='symm', mode='same').flatten()
    X_out = np.hstack((X,X_filt))
    return X_out

##################### ROTATION ########################
from scipy.ndimage.interpolation import rotate
def Rotation(image):
    n_samples,n_features=image.shape
    imRot90=np.zeros((n_samples,n_features))
    #imRot180=np.zeros((n_samples,n_features))
    #imRot270=np.zeros((n_samples,n_features))
    for i in range(image.shape[0]):
        Im=image[i].reshape(28,28)
        Im90=rotate(Im,90)
        #Im180=rotate(Im,180)
        #Im270=rotate(Im,270)
        
        im90=Im90.reshape(784)
        #im180=Im180.reshape(784)
        #im270=Im270.reshape(784)
        
        imRot90[i]=im90
    #imRot180[i]=im180
    #imRot270[i]=im270
    #digit=np.vstack((image,imRot90,imRot180,imRot270))
    digit=np.vstack((image,imRot90))
    return digit

##################### Binerization ########################
def thresh_images(X,k=0.25):
    X_thresh = X.copy()
    for i in range(X.shape[0]):
        X_thresh[i,:] = X_thresh[i,:] > k*np.max(X_thresh[i,:])
    return X_thresh

##################### Binerization ########################
def normalization(X):
    # return (X-np.mean(X,axis=0))/np.std(X,axis=0)
    return (X-np.mean(X,axis=0))/np.std(X,axis=0)

##################### SCaling ########################
def scale(X):
    return (X - np.min(X, axis = 1)[:,np.newaxis])/(np.max(X, axis = 1) - np.min(X, axis = 1))[:,np.newaxis]

##################### image transpose ########################
def transpose_images(X):
    X_trans = X.copy()
    for i in range(X.shape[0]):
        X_trans[i,:] = X[i,:].reshape(( 28, 28)).T.flatten()
    return X_trans

##################### Dataset translation ########################
def translate_img(img):
    img2 = np.zeros(img.shape)
    k = [1,2,3][np.random.randint(0,3)]
    lr = np.random.randint(0,2) #0 left 1 rigth
    ud = np.random.randint(0,2) #0 down 1 left
    if (lr,ud) == (0,0):
        img2[k:,:img.shape[1]-k] = img[:img.shape[0]-k,k:]
    elif (lr,ud) == (0,1):
        img2[:img.shape[0]-k,:img.shape[1]-k] = img[k:,k:]
    elif (lr,ud) == (1,0):
        img2[k:,k:] = img[:img.shape[0]-k,:img.shape[1]-k]
    else:
        img2[:img.shape[0]-k,k:] = img[k:,:img.shape[1]-k]
    return img2

def double_images(X,y):
    n,m = X.shape
    y_double = np.array([y,y]).flatten()
    X_double = np.zeros([2*n, m])
    X_double[:n,:] = X
    for i in range(n):
        X_double[i+n,:] = translate_img(X_double[i,:].reshape(( 28, 28))).flatten()
    return X_double, y_double