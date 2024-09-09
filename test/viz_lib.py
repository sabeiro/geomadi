#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)


def rawData(X,xlab="hour",ylab="location"):
    colN = 20
    f, ax = plt.subplots(ncols=colN,nrows=2)#,sharex=True,sharey=True)
    f.tight_layout()
    f.subplots_adjust(.1,.1,.9,.9,0,0)
    N = int(X.shape[0]/(colN*2))
    for i,a in enumerate(ax):
        for j,b in enumerate(a):
            k = i + j*2
            b.imshow(X[k*N:(k+1)*N,:])
            if i == 1:
                b.set_xlabel("hour")
            else :
                b.set_xticks([])
            if j == 0:
                b.set_ylabel("cell")
            else :
                b.set_yticks([])
    plt.show()
