"""
train_metric:
calculate important metrics for scoring and future predictions.
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
import geomadi.train_modelList as modL
import statsmodels.api as sm
import sklearn as sk
from scipy.optimize import leastsq as least_squares

def mutualInformation(x,y):
    """calculate mutual information"""
    sum_mi = 0.0
    x_value_list = np.unique(x)
    y_value_list = np.unique(y)
    Px = np.array([ len(x[x==xval])/float(len(x)) for xval in x_value_list ]) #P(x)
    Py = np.array([ len(y[y==yval])/float(len(y)) for yval in y_value_list ]) #P(y)
    for i in xrange(len(x_value_list)):
        if Px[i] ==0.:
            continue
        sy = y[x == x_value_list[i]]
        if len(sy)== 0:
            continue
        pxy = np.array([len(sy[sy==yval])/float(len(y))  for yval in y_value_list]) #p(x,y)
        t = pxy[Py>0.]/Py[Py>0.] /Px[i] # log(P(x,y)/( P(x)*P(y))
        sum_mi += sum(pxy[t>0]*np.log2( t[t>0]) ) # sum ( P(x,y)* log(P(x,y)/( P(x)*P(y)) )
    return sum_mi

def linLeastSq(X,y):
    """linear model with least square"""
    tml = modL.modelList()
    clf = tml.regL['elastic_cv']['mod']
    model = clf.fit(X,y)
    return model.coef_
    # model = sm.OLS(y,X).fit()
    # return model.params
    if False:
        predictions = model.predict(X)
        X1 = np.c_[X,np.ones(X.shape[0])] # add bias term
        beta_hat = np.linalg.lstsq(X1,y)[0][:X.shape[1]]
        return beta_hat
        beta_hat = np.dot(np.linalg.inv(np.dot(X1.T, X1)), np.dot(X1.T, y))
        beta_hat = np.linalg.lstsq(np.vstack([X, np.ones(len(X))]).T, y)[0]
    def ser_sin(x,t,param):
        return x*t.sum(axis=0)
    def ser_fun_min(x,t,y,param):
        return ser_sin(x,t,param).sum() - y.sum()
    x0 = X.sum(axis=0)
    x0 = x0/x0.mean()
    x0 = np.linspace(1,1,X.shape[1])
    res = least_squares(ser_fun_min,x0,args=(X,y,x0))
    beta_hat = res['x']
    return beta_hat

def linWeight(X,y,n_source=5,isPlot=False):
    """performs a linear weighting using n_source features"""
    if isPlot:
        n = X.shape[1]
        r = np.corrcoef(X.T).sum(axis=1)/n
        s = X.sum(axis=0)
        x = X.sum(axis=1)
        plt.imshow(X.T)
        plt.show()
    beta_hat = linLeastSq(X,y)
    if len(beta_hat) <= n_source:
        return beta_hat, n_source
    idx = pd.DataFrame({"c":beta_hat,"use":beta_hat > np.percentile(beta_hat,100.-n_source/len(beta_hat)*100),"beta_hat":0})
    X1 = X[:,idx['use']]
    c = linLeastSq(X1,y)
    idx.loc[idx['use'],"beta_hat"] = c
    if isPlot:
        X2 = np.multiply(X,beta_hat[:,np.newaxis].T)
        x2 = X2.sum(axis=1)
        X3 = np.multiply(X1,c[:,np.newaxis].T)
        x3 = X3.sum(axis=1)
        plt.title("corr_w %.2f corr_o %.2f dif_w %.2f dif_o %.2f" % (
            sp.stats.pearsonr(y,x2)[0]
            ,sp.stats.pearsonr(y,x3)[0]
            ,(y.sum()-x2.sum())/(y.sum()+x2.sum())
            ,(y.sum()-x3.sum())/(y.sum()+x3.sum()))
        )
        plt.plot(y,label="ref")
        plt.plot(x2,label="weighted")
        plt.plot(x3,label="optimized")
        plt.plot(x,label="raw")
        plt.legend()
        plt.show()
    return idx['beta_hat'].values, n_source

def linWeightOpt(X,y,n_source=5,isPlot=False):
    """performs a linear weighting optimizing on the number of sources until the minimum n_source"""
    beta_hat = linLeastSq(X,y)
    scorL = []
    idx = pd.DataFrame({"c":beta_hat,"use":True,"beta_hat":beta_hat})
    for i in range(X.shape[1]-n_source):
        setL = idx.loc[idx['use'],'beta_hat'] <= min(idx.loc[idx['use'],'beta_hat'])
        setL = setL[setL]
        idx.loc[setL.index,"use"] = False
        X1 = X[:,idx['use']]
        if X1.shape[1] == 0:
            break
        beta = linLeastSq(X1,y)
        idx.loc[~idx['use'],"beta_hat"] = 0
        idx.loc[idx['use'],"beta_hat"] = beta
        beta = idx['beta_hat'].values
        x1 = np.multiply(X,beta_hat).sum(axis=1)
        mean = np.mean(list(x1)+list(y))
        rmse = np.sqrt((x1-y)**2)
        rmse = rmse.sum()/len(y)/mean
        r = sp.stats.pearsonr(x1,y)[0]
        scorL.append({"r":r,"beta_hat":beta_hat,"n":sum(idx['use']),"m":rmse})

    beta = beta_hat
    maxV = -float('inf')
    n_cell = X.shape[1]
    for i,v in enumerate(scorL):
        s = v['r']/np.log(v['n']+1.)/v['m']
        if s > maxV:
            beta = v['beta_hat']
            maxV = s
            n_cell = v['n']
    if False:
        X2 = np.multiply(X,idx['c'].values[:,np.newaxis].T)
        x2 = X2.sum(axis=1)
        X3 = np.multiply(X,beta[:,np.newaxis].T)
        x3 = X3.sum(axis=1)
        x = X.sum(axis=1)
        print("corr %.2f -> %.2f -> %.2f cells %d" % (sp.stats.pearsonr(y,x)[0]
                                                      ,sp.stats.pearsonr(y,x2)[0]
                                              ,sp.stats.pearsonr(y,x3)[0],n_cell))
    if isPlot:
        tL = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in hL]
        X2 = np.multiply(X,idx['c'].values[:,np.newaxis].T)
        x2 = X2.sum(axis=1)
        X3 = np.multiply(X,beta[:,np.newaxis].T)
        x3 = X3.sum(axis=1)
        x = X.sum(axis=1)
        x = x*y.sum()/x.sum()
        plt.title("poi %s corr_r %.2f corr_o %.2f" % (g.iloc[0][idField]
            ,sp.stats.pearsonr(y,x)[0]
            ,sp.stats.pearsonr(y,x3)[0]))
        plt.plot(tL,y,label="ref")
        plt.plot(tL,x2,label="selected")
        plt.plot(tL,x3,label="optimized")
        plt.plot(tL,x,label="raw")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()
    return beta, n_cell






