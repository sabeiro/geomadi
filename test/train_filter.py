#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)


def binOutlier(y,nBin=6,threshold=3.5):
    """bin with special treatment for the outliers"""
    n = nBin
    ybin = [threshold] + [x*100./float(n-1) for x in range(1,n-1)] + [100.-threshold]
    pbin = np.unique(np.nanpercentile(y,ybin))
    n = min(n,pbin.shape[0])
    delta = (pbin[n-1]-pbin[0])/float(n-1)
    pbin = [np.nanmin(y).min()] + [x*delta + pbin[0] for x in range(n)] + [np.nanmax(y).max()]
    if False:
        plt.hist(y,fill=False,color="red")
        plt.hist(y,fill=False,bins=pbin,color="blue")
        plt.show()
        sigma = np.std(y) - np.mean(y)
    t = np.array(pd.cut(y,bins=np.unique(pbin),labels=range(len(np.unique(pbin))-1),right=True,include_lowest=True))
    t[np.isnan(t)] = -1
    t = np.asarray(t,dtype=int)
    return t, pbin

def binMatrix(X,nBin=6,threshold=2.5):
    """bin a continuum parametric matrix"""
    c_M = pd.DataFrame()
    psum = pd.DataFrame(index=range(nBin+2))
    for i in range(X.shape[1]):
        xcol = X[:,i]
        c_M.loc[:,i], binN = binOutlier(xcol,threshold=2.5)
        psum.loc[range(len(binN)),i] = binN
    return c_M, psum
    
