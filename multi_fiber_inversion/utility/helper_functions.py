import numpy as np
from scipy.special import erfi, legendre, gamma, sph_harm
from scipy.optimize import brentq

def cart2sphere(x):
    ## Note: theta, phi convention here is flipped compared to dipy.core.geometry.cart2sphere
    r = np.sqrt(x[:,0]**2 + x[:,1]**2 + x[:,2]**2)
    theta = np.arctan2(x[:,1], x[:,0])
    phi = np.arccos(x[:,2]/r)
    return np.column_stack([theta, phi])

def sphere2cart(x):
    theta = x[:,0]
    phi = x[:,1]
    xx = np.sin(phi)*np.cos(theta)
    yy = np.sin(phi)*np.sin(theta)
    zz = np.cos(phi)
    return np.column_stack([xx, yy, zz]) 

def S2hemisphere(x):
    x_copy = np.copy(x)
    x_polar = cart2sphere(x_copy)
    ix = np.argwhere(x_polar[:,1] > np.pi/2).ravel()
    x_copy[ix, :] = -1*x_copy[ix, :] 
    return x_copy

def get_odf_transformation(n):
    T = np.zeros((len(n), len(n)))
    for i in range(T.shape[0]):
        P_n = legendre(n[i])
        T[i, i] = P_n(0)
    return T
