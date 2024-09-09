"""
train_shape:
reduce curves into their main statistical properties to provide an accurate and minimal learning set.
"""

import random, json, datetime, re
from scipy import signal
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import geomadi.series_stat as s_s

def periodic(X,period=24,isPlot=False):
    """feature selection of signals over period"""
    n_per = int(X.shape[1]/period)
    X_d = np.zeros(shape=(X.shape[0],period))
    for i in range(n_per):
        X_d = X_d + X[:,i*period:(i+1)*period]
    def ser_fun(x,t,param):
        #return x[0]*np.exp(-x[1]*(t-x[2])**2) + x[3]*np.exp(-x[4]*(t-x[5])**2)
        return x[0] + x[1] * t + x[2] * t * t
    def ser_fun_min(x,t,y,param):
        return ser_fun(x,t,param) - y
    param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
    rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
    X_d = X_d / rsum[:,np.newaxis]
    X_d[np.isnan(X_d)] = 0
    t = np.array(range(X_d.shape[1])).astype('float')
    x0 = [2.29344311e-02,4.91473902e-03,-1.56686399e+01]#,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
    param = [2.*np.pi/(18.),2.*np.pi/(7.*18.)]
    i = 606
    convL = []
    for i in range(X_d.shape[0]):
        res = least_squares(ser_fun_min,x0,args=(t,X_d[i],param))
        convL.append(res.x)
    if isPlot:
        X = np.array(sact[hL])
        X = np.array(vist[hL])
        X_d = np.zeros(shape=(X.shape[0],period))
        n_per = int(X.shape[1]/period)
        for i in range(n_per):
            X_d = X_d + X[:,i*period:(i+1)*period]
        X_d = X_d.mean(axis=0)
        X_d = X_d/sum(X_d)
        t = np.array(range(X_d.shape[0])).astype('float') + 11
        res = least_squares(ser_fun_min,x0,args=(t,X_d,param))
        plt.plot(t,X_d,label="original")
        plt.plot(t,ser_fun(res.x,t,param),label="parab fit")
        plt.legend()
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.show()
    return pd.DataFrame(convL,columns=["t_interc","t_slope","t_conv"])

def daily(X,period=24,isPlot=False):
    """feature selection of signals over period"""
    n_per = int(X.shape[1]/period)
    X1 = X[:,:period*n_per]
    X_d = X1.reshape(-1,n_per,period).sum(axis=2)
    x0 = [2.29344311e-02,4.91473902e-03,1.56686399e+01,-2.52343576e-04,2.91360123e-06,-9.72000000e-04]
    param = [2.*np.pi/(7),2.*np.pi/(7.*4)]
    def ser_fun(x,t,param):
        #return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
        return x[0] + x[1] * np.sin(param[0]*t + x[2]) + x[3]*t + x[4]*t*t
    def ser_fun_min(x,t,y,param):
        return ser_fun(x,t,param) - y

    rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
    X_d = X_d / rsum[:,np.newaxis]
    X_d[np.isnan(X_d)] = 0
    t = np.array(range(X_d.shape[1])).astype('float')
    convL = []
    for i in range(X_d.shape[0]):
        res = least_squares(ser_fun_min,x0,args=(t,X_d[i],param))
        convL.append(res.x)
    if isPlot:
        X = np.array(sact[hL])
        X = np.array(vist[hL])
        n_per = int(X.shape[1]/period)
        X1 = X[:,:period*n_per]
        X_d = X1.reshape(-1,n_per,period).sum(axis=2)
        X_d = X_d.mean(axis=0)
        # rsum = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X_d)
        # X_d = X_d / rsum[:,np.newaxis]
        t = np.array(range(X_d.shape[0])).astype('float')
        res = least_squares(ser_fun_min,x0,args=(t,X_d,param))
        plt.plot(t,X_d,label="original")
        plt.plot(t,ser_fun(x1,t,param),label="complete fit")
        plt.plot(t,ser_fun([x1[0],x1[1],x1[2],0.,0.],t,param),label="sinus part")
        plt.plot(t,ser_fun([x1[0],0.,0.,x1[3],x1[4]],t,param),label="parab part")
        plt.legend()
        plt.xlabel("day")
        plt.ylabel("count")
        plt.show()
    return pd.DataFrame(convL,columns=["t_interc","t_slope","t_conv"])

def seasonal(X,period=18,isPlot=False):
    """feature selection of signals over season"""
    param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
    x1 = [-6.03324843e+02,1.25552574e+02,-3.98806217e+00,1.67340000e-02,1.98480000e-02,9.72000000e-04]
    def ser_sin(x,t,param):
        #return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
        return x[0] + x[1] * np.sin(param[0]*t + x[2]) + x[3]*t + x[4]*t*t
        
    def ser_sin_min(x,t,y,param):
        return ser_sin(x,t,param) - y
    t = np.array(range(X.shape[1])).astype('float')
    convL = []
    for i in range(X.shape[0]):
        res = least_squares(ser_sin_min,x1,args=(t,X[i],param))
        convL.append([res.x[1],res.x[3],res.x[4]])
    if isPlot:
        X = np.array(sact[hL])
        X = np.array(vist[hL])
        X_d = X.mean(axis=0)
        t = np.array(range(X_d.shape[0])).astype('float')
        res = least_squares(ser_sin_min,x1,args=(t,X_d,param))
        plt.plot(t,X_d,label="original")
        plt.plot(t,ser_sin(res.x,t,param),label="sinus + parab fit")
        plt.legend()
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.show()
    return pd.DataFrame(convL,columns=["t_ampl","t_trend1","t_trend2"])

def monthly(X,period=24):
    """feature selection of signals over season"""
    param = [2.*np.pi/(period),2.*np.pi/(7.*period)]
    def ser_sin(x,t,param):
        return x[0] + x[1] * np.sin(param[0]*t + x[2])*(1 + x[3]*np.sin(param[1]*t + x[4]))
    def ser_sin_min(x,t,y,param):
        return ser_sin(x,t,param) - y
    t = np.array(range(X.shape[1])).astype('float')
    convL = []
    x1 = [-6.03324843e+02,1.25552574e+02,-3.98806217e+00,1.67340000e-02,1.98480000e-02,9.72000000e-04]
    for i in range(X.shape[0]):
        res = least_squares(ser_sin_min,x1,args=(t,X[i],param))
        convL.append([res.x[0],res.x[1],res.x[3]])
    if False:
        i = 0
        i = 1188
        t = np.array(range(X.shape[1])).astype('float')
        res = least_squares(ser_sin_min,x1,args=(t,X[i],param))
        plt.plot(t,X[i],label="original")
        plt.plot(t,ser_sin(res.x,t,param),label="sinus + parab fit")
        plt.plot(t,ser_sin([x1[0],x1[1],x1[2],0.,0.],t,param),label="sinus part")
        plt.plot(t,ser_sin([x1[0],0.,0.,x1[3],x1[4]],t,param),label="parab part")
        plt.plot(t,ser_sin([x1[0],x1[1],x1[2],x1[3],x1[4]],t,param),label="complete fit")
        plt.legend()
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.show()
        
    return pd.DataFrame(convL,columns=["t_m_inter","t_m_trend1","t_m_trend2"])

def statistical(X):
    """calculate basic statistical properties"""
    t_M = pd.DataFrame()
    t_M.loc[:,'t_max'] = np.max(X,axis=1)
    t_M.loc[:,'t_std'] = np.std(X,axis=1)
    t_M.loc[:,'t_sum'] = np.nansum(X,axis=1)
    t_M.loc[:,'t_median'] = X.argmax(axis=1)
    return t_M

def calcPCA(X):
    """return PCA matrix"""
    pca = sk.decomposition.PCA().fit(X)
    return pca.transform(X)
       
def smooth(X,width=3,steps=5):
    """smooth signal"""
    return np.apply_along_axis(lambda y: s_s.serSmooth(y,width=width,steps=steps),axis=1,arr=X)

def runAv(X,steps=5):
    """running average"""
    return np.apply_along_axis(lambda y: s_s.serRunAv(y,steps=steps),axis=1,arr=X)

def interpMissing(X):
    """interpolate missing values"""
    return np.apply_along_axis(lambda y: s_s.interpMissing(y),axis=1,arr=X)

##https://stackoverflow.com/questions/41860817/hyperparameter-optimization-for-deep-learning-structures-using-bayesian-optimiza
