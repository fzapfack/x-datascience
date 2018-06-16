import numpy as np

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian(x,y,sigma=0.02):
        return np.exp(-np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2)))

def gaussian_kernel(x, y, sigma=0.5):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))