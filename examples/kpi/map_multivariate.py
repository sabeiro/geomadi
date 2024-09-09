##%pylab inline
##http://scikit-learn.org/stable/modules/ensemble.html
##https://towardsdatascience.com/bayesian-linear-regression-in-python-using-machine-learning-to-predict-student-grades-part-2-b72059a8ac7e
import os, sys, gzip, random, csv, json, datetime,re
import time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_lib as tlib
import geomadi.train_shapeLib as shl
import geomadi.train_filter as t_f
import custom.lib_custom as l_t
import geomadi.train_execute as t_e
import importlib
from io import StringIO
from sklearn import decomposition
from sklearn.decomposition import FastICA, PCA

def plog(text):
    print(text)

plog('-------------------load/def------------------------')
ops = {"isScore":True,"lowcount":True,"p_sum":True,"isWeekday":True}
fSux = "20"
idField = "id_poi"
custD = "mc"
if len(sys.argv) > 1:
    fSux = sys.argv[1]

poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
mapL = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")

mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
mist = mist.replace(float('nan'),0.)
mist = mist.groupby("id_poi").agg(sum).reset_index()
hL1 = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]

sact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac.csv.gz",compression="gzip")
sact = pd.merge(mapL,sact,on="cilac",how="left")
sact = sact.replace(float('nan'),0.)
hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
del sact[hL[-1]]
hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
hL = sorted(list(set(hL) & set(hL1)))

import importlib
importlib.reload(t_e)

if ops["isScore"]:
    act = t_e.joinSource(sact,mist,how="inner",idField=idField)
    scorM1 = t_e.scorPerf(act,step="etl",idField=idField)

thVal = []
keepL = []
sact.loc[:,"weight"] = 0.
for i,g in sact.groupby(idField):
    r = g[hL].T.corr().sum(axis=1) #remove off distribution cells
    r = r[r.values > g.shape[0]/3.]
    keepL.append(list(r.index))
    setL = mist[idField] == i
    if sum(setL) == 0:
        continue
    g = g.loc[r.index]
    if False:
        tlib.plotPairGrid(g[hL].T)
    X = g[hL].values.T
    y = mist[mist[idField] == i][hL].values[0]
    X1 = np.c_[X,np.ones(X.shape[0])] # add bias term
    beta_hat = np.linalg.lstsq(X1,y)[0][:X.shape[1]]
    sact.loc[g.index,"weight"] = beta_hat
    thL = np.linspace(0,1,20)
    distL = [abs(sum(X[:,beta_hat>i].sum(axis=1)-y)) for i in thL]
    thM = [i for i,x in enumerate(distL) if x == min(distL)]
    thVal.append(thL[thM][0])

if False:
    keepL = [item for sublist in keepL for item in sublist]
    sact = sact[sact.index.isin(keepL)]
    print(np.mean(thVal))
    
sact = sact.sort_values([idField,'weight']).reset_index()
print(sact[sact['weight'] > 0.].shape,sact.shape)

weightL = pd.DataFrame({"weight":sact['weight'].values},index=sact['id_poi'])
refS = mist[hL].sum(axis=1)
refS.index = mist['id_poi']
actS = sact[[idField]+hL].groupby(idField).agg(np.sum).sum(axis=1)
actD = pd.DataFrame({"act":actS,"ref":refS})
actD = actD.replace(float('nan'),1.)
actD.loc[:,"cor"] = actD['act']/actD['ref']
weightL.loc[:,"cor"] = actD['cor']
weightL.loc[:,"weight_cor"] = weightL['weight']*weightL['cor']
tact = sact.copy()
tact.loc[:,hL] = np.multiply(sact[hL].values,weightL['weight_cor'].values[:,np.newaxis])
tact = tact[[idField] + hL].groupby(idField).agg(np.sum).reset_index()
tact.loc[:,"id_poi"] = tact['id_poi'].astype(int)
#tact.loc[:,hL] = tact.loc[:,hL] - tact.loc[:,hL].min().min()

pact = t_e.joinSource(sact,mist,how="inner",idField=idField)
gact = t_e.joinSource(tact,mist,how="inner",idField=idField)

if False:
    t_e.plotSum(pact)
    t_e.plotSum(tact)

if ops['isScore']:
    scorM2 = t_e.scorPerf(act,step="reg",idField=idField)
    
if ops['isScore']:
    scor = pd.merge(scorM1,scorM2,on="id_poi",how="outer")
    # for i in [scorM3,scorM4]:
    #     scor = pd.merge(scor,i,on="id_poi",how="outer")
    scor.to_csv(baseDir + "raw/"+custD+"/scorMap.csv")

gact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip",index=False)







if False:
    ica = FastICA(n_components=3)
    S_ = ica.fit_transform(X)  # Reconstruct signals
    A_ = ica.mixing_  # Get estimated mixing matrix

    # For comparison, compute PCA
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

    
    dec = decomposition.FastICA(n_components=X.shape[0], whiten=True)
    dec.fit(X)
    plt.imshow(dec.components_)
    plt.show()


    
    featL = tlib.featureRelevanceTree(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=0)
    shA_X = shared(X_train)
    linear_model = pm.Model()
    with linear_model:
        alpha = pm.Normal("alpha", mu=y_train.mean(),sd=10)
        betas = pm.Normal("betas", mu=0, sd=1000, shape=X.shape[1])
        sigma = pm.HalfNormal("sigma", sd=100) 
        mu = alpha + np.array([betas[j]*shA_X[:,j] for j in range(X.shape[1])]).sum()
        #    mu = alpha + pm.dot(betas, X_train.T)
        likelihood = pm.Normal("likelihood", mu=mu, sd=sigma, observed=y_train)
        #    map_estimate = pm.find_MAP(model=linear_model, fmin=optimize.fmin_powell)
        step = pm.NUTS()
        trace = pm.sample(1000, step)
        
    pm.traceplot(trace);
    plt.show()

    
    plt.imshow(X.T.corr())
    plt.show()

    # Context for the model
    with pm.Model() as normal_model:
        formula = ""
        family = pm.glm.families.Normal()
        pm.GLM.from_formula(formula,data=X_train,family=family)
        normal_trace = pm.sample(draws=2000, chains = 2, tune = 500, njobs=-1)

if False:
        
    df = pd.DataFrame({"x":X[:,0],"y":y})
    from scipy.optimize import fmin_powell
    with pm.Model() as mdl_ols:
        pm.glm.glm('y ~ 1 + x', df, family=pm.glm.families.Normal())
        start_MAP = pm.find_MAP(fmin=fmin_powell, disp=True)
        trc_ols = pm.sample(2000, start=start_MAP, step=pm.NUTS())

    ax = pm.traceplot(trc_ols[-1000:], figsize=(12,len(trc_ols.varnames)*1.5),lines={k: v['mean'] for k, v in pm.df_summary(trc_ols[-1000:]).iterrows()})

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(sact,la,test_size=0.25,random_state=42)


    intercept = pm.Normal('Intercept', mu = 0, sd = 10)
    slope = pm.Normal('slope', mu = 0, sd = 10)
    sigma = pm.HalfNormal('sigma', sd = 10)
    mean = intercept + slope * X.loc[0:499, 'Duration']
    Y_obs = pm.Normal('Y_obs', mu = mean, sd = sigma, observed = y.values[0:500])
    step = pm.NUTS()
    linear_trace_500 = pm.sample(1000, step)
    
    
    print(sklearn.metrics.mutual_info_score(X[0],y))
    cline = np.apply_along_axis(lambda x: sklearn.metrics.mutual_info_score(x,y),axis=1,arr=X)
    ##ordinary least square regression
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X, X.T)), X), y)



