"""
train_interp:
interpolation utils for time series
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra

def interpDoubleGauss(X,t,y,x0=[None],p0=[None]):
    """interpolate via double Gaussian"""
    if not any(x0):
        x0 = [3.,3.,min(y)]
        x0[0] = 1./(2.*np.power(x0[0],2.))
        x0[1] = 1./(2.*np.power(x0[1],2.))
    if not any(p0):
        p0 = [8.5,13.5,max(y)*.75,max(y)*.9]
        # p0[2] = p0[2]/(np.sqrt(2.*np.pi)*x0[0])
        # p0[3] = p0[3]/(np.sqrt(2.*np.pi)*x0[1])
    def ser_dExp(x,t,p):
        exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
        exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
        return x[2] + p[2]*exp1 + p[3]*exp2
    def ser_residual(x,t,y,p):
        return (y-ser_dExp(x,t,p))
    res = least_squares(ser_residual,x0,args=(t,y,p0),method="trf",loss="soft_l1")#,bounds=(0,200.))
    #res = least_squares(ser_residual,x0,args=(t,y,p0))
    x0 = res.x
    def ser_dExp(p,t,x):
        exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
        exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
        return x[2] + p[2]*exp1 + p[3]*exp2
    def ser_residual(p,t,y,x):
        return (y-ser_dExp(p,t,x))
    res = least_squares(ser_residual,p0,args=(t,y,x0),method="trf",loss="soft_l1")#,bounds=(0,200.))
    #res = least_squares(ser_residual,p0,args=(t,y,x0))
    p0 = res.x
    t1 = t#np.linspace(t[0],t[len(t)-1],50)
    y1 = ser_dExp(p0,t1,x0)
    return t1, y1, p0, x0

def interpPoly(X,t,y,x0=[None],p0=[None]):
    """interpolate via polynomial"""
    if not any(x0):
        x0 = [1.50208343e+01,-5.06158347e+00,3.59972679e-01,2.50569233e-01,-2.91416175e-02,1.09982221e-03,-1.37937148e-05]
        p0 = [0]
    def ser_dExp(x,t,p):
        res1 = x[0] + t*x[1] + t*t*x[2] + t*t*t*x[3] + t**4*x[4] + t**5*x[5] + t**6*x[6]
        return res1
    def ser_residual(x,t,y,p):
        return (y-ser_dExp(x,t,p))
    res = least_squares(ser_residual,x0,args=(t,y,p0))
    x0 = res.x
    t1 = t#np.linspace(t[0],t[len(t)-1],50)
    y1 = ser_dExp(x0,t1,00)
    return t1, y1, p0, x0

def interpFun(X,t,y,how="gauss"):
    """interpolate via fitting function"""
    if how == "gauss":
        t,y1,p1,x1 = interpDoubleGauss(t,y)
    else :
        t,y1,p1,x1 = interpPoly(t,y)
    if False:
        def ser_dExp(x,t,p):
            exp1 = np.exp(-np.power(t - p[0], 2.)*x[0])
            exp2 = np.exp(-np.power(t - p[1], 2.)*x[1])
            return x[2] + p[2]*exp1 + p[3]*exp2
        pt = p1.copy()
        pt[2] = 0
        y3 = ser_dExp(x1,t,pt)
        pt = p1.copy()
        pt[3] = 0
        y4 = ser_dExp(x1,t,pt)
        print(p1)
        print(x2)
        plt.plot(t,y,label="vist")
        plt.plot(t,y1,label="ref_gauss")
        plt.plot(t,y3,label="gauss_1")
        plt.plot(t,y4,label="gauss_2")
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.legend()
        plt.show()
    return y1 #np.concatenate([p1,x1])

def fit(X,how="gauss"):
    t = np.array([int(x) for x in range(X.shape[1])])
    X1 = np.apply_along_axis(lambda y: interpFun(t,y,how),axis=1,arr=X)
    return X1
    
def replaceOffChi(X,idL,threshold=0.003,dayL=False,isPlot=False):
    """replace values with bad chi square score"""
    df = pd.DataFrame(X)
    hL = df.columns
    df.loc[:,"id"] = idL
    avid = df.groupby("id").agg(np.mean).reset_index()
    _, chiSq = sp.stats.chisquare(X.T)
    chiSq = np.nan_to_num(chiSq,0.)
    if isPlot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(chiSq,bins=20)
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlabel("p-value")
        plt.ylabel("counts")
        plt.title("p-value distribution per location/day")
        plt.show()
    setL = chiSq > threshold
    if any(dayL == False):
        df.loc[:,"day"] = dayL
        dayN = np.unique(df[setL]['day'],return_counts=True)
        dayN = pd.DataFrame({"day":dayN[0],"count":dayN[1]})
        print(dayN[dayN['count']>50])
    dayN = np.unique(df[setL]['id'],return_counts=True)
    dayN = pd.DataFrame({"id":dayN[0],"count":dayN[1]})
    print("replacing %d noisy points" % sum(setL))
    clusL = df[setL]['id']
    clusI = [avid[avid['id'] == x].index.values[0] for x in clusL]
    if isPlot:
        X1 = X[setL]
        X2 = avid.iloc[clusI][hL].values
        plt.figure()
        plt.title("image representation of time series")
        N = X1.shape[0]
        j = random.sample(range(N),N)
        for i in range(6):
            ax = plt.subplot(3,4,i*2+1)
            plt.plot(X1[j[i]])
            ax = plt.subplot(3,4,i*2+2)
            plt.plot(X2[j[i]])
        plt.show()
        
        X[setL] = avid.iloc[clusI][hL].values
    return dayN
        

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
