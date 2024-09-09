import random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy import signal as sg
        
def gkern(nlen=21,nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(nlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., nlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gkern2(nlen=21,nsig=3):
    """Returns a 2D Gaussian kernel array."""
    inp = np.zeros((nlen, nlen))
    inp[nlen//2, nlen//2] = 1
    return fi.gaussian_filter(inp, nsig)

def gkern_norm(l=5,sig=1.):
    """creates gaussian kernel with side length l and a sigma of sig """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)        
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)

def disk(nlen=21,nsig=3):
    X = gkern(nlen=nlen,nsig=nsig)
    return X > X.mean()
