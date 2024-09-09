"""
train_score:
calculate scores for model training
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
import geomadi.series_stat as s_s
import sklearn as sk
from scipy.optimize import leastsq as least_squares

def relErr(y1,y2):
    """compute the relative error"""
    mean = np.mean(list(y1)+list(y2))
    rmse = np.sqrt((y1-y2)**2).sum()/len(y1)
    err = rmse/mean
    return mean, rmse, err

def calcMetrics(y1,y2):
    # lag = np.argmax(sp.signal.correlate(y1,y2))
    # c_sig = np.roll(y2, shift=int(np.ceil(lag)))
    # t = sp.signal.csd(y1,y2)
    # t = sp.signal.coherence(y1,y2)
    mean, rmse, err = relErr(y1,y2)
    return {
        "cor":sp.stats.pearsonr(y1,y2)[0]
        ,"rel_err":err
        ,"mut_info":sk.metrics.mutual_info_score(y1,y2)
        ,"dif":2.*(np.sum(y1)-np.sum(y2))/(np.sum(y1)+np.sum(y2))
    }

def scorPerf(act,step="input",idField="id_poi",col1="act",col2="ref"):
    if act.shape[0] == 0:
        print('empty dataframe')
        return pd.DataFrame()
    def clampF(x):
        return pd.Series({
            "r_"+step:sp.stats.pearsonr(x[col1],x[col2])[0]
            ,"d_"+step:(x[col1].sum()-x[col2].sum())/(x[col1].sum()+x[col2].sum())
            ,"v_"+step:relErr(x[col1],x[col2])[2]
            ,"s_"+step:x[col2].sum()
        })
    scorM = act.groupby([idField]).apply(clampF).reset_index()
    print("score %s: %.2f" % (step,scorM['r_'+step].mean()) )
    return scorM

def scorStat(X,y):
    """return a scoring based on correlation and regression"""
    X, y = np.array(X), np.array(y)
    X = np.apply_along_axis(lambda y: s_s.interpMissing(y),axis=1,arr=X)
    y = s_s.interpMissing(y)
    clf = sk.linear_model.BayesianRidge() #.RidgeCV(alphas=[0.1, 1.0, 10.0]).MultiTaskElasticNet().LogisticRegression().HuberRegressor()
    lr = sk.linear_model.LinearRegression()
    sline = np.apply_along_axis(lambda x: np.nansum(x),axis=1,arr=X)
    #chis = np.apply_along_axis(lambda x: sp.stats.chisquare(x,pline),axis=1,arr=X)[:,0]
    return {
        "s_lin":lr.fit(X.T,y).coef_
        ,"s_ent":np.apply_along_axis(lambda x: sk.metrics.mutual_info_score(x,y),axis=1,arr=X)
        ,"s_cor":np.apply_along_axis(lambda x: sp.stats.pearsonr(x,y),axis=1,arr=X)[:,0]
        ,"s_rid":clf.fit(X.transpose(),y).coef_
        ,"s_dif":abs(np.nansum(X,axis=1).astype('float') - y.sum())/y.sum()
        #,"s_sqr":np.linalg.lstsq(X.T,y)[0]
    }

def score(X1,X2,hL=[],idField="id_poi",idFeat="id_feat"):
    """return a scoring based on correlation and regression"""
    missingL = []
    scorM = []
    for i,x2 in X2.iterrows():
        selI = X1[idField] == x2[idField]            
        if any(selI) == False:
            missingL.append(x2[idField])
            continue
        X = np.array(X1.loc[selI,hL])#.transpose()
        y = np.array(x2.loc[hL])
        if (sum(y) <= 0.):
            missingL.append(x2[idField])
            continue
        s = scorStat(X,y)
        s[idField] = X1.loc[selI,idField].values[0]
        s[idFeat] = X1.loc[selI,idFeat]
        scorM.append(pd.DataFrame(s))
    scorM = pd.concat(scorM)
    return scorM, missingL#.replace(np.nan,0)


