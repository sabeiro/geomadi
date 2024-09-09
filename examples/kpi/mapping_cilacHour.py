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
import geomadi.series_lib as s_l
import geomadi.geo_enrich as g_e
import importlib
from io import StringIO
from sklearn import decomposition
from sklearn.decomposition import FastICA, PCA
import pymongo
import osmnx as ox
import shapely as sh

def plog(text):
    print(text)

plog('-------------------load/def------------------------')
ops = {"isScore":True,"lowcount":True,"p_sum":True,"isWeekday":True,"isType":False}
fSux = "20"
idField = "id_poi"
custD = "tank"
cred = json.load(open(baseDir + '/credenza/geomadi.json'))
metr = json.load(open(baseDir + '/raw/basics/metrics.json'))['metrics']
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

#idField = "id_clust"
#custD = "tank"
##Spessart Süd, GB - 1276

if len(sys.argv) > 1:
    fSux = sys.argv[1]

poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
if custD == "tank":
    poi = poi[poi['use'] == 3]

mapL  = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL = dateL[dateL['use'] > 0]
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_h.csv.gz",compression="gzip")
mist = mist.replace(float('nan'),0.)
mist = mist.groupby(idField).agg(sum).reset_index()
mist.loc[:,idField] = mist[idField].astype(str)
hL1 = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]

sact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac/tank_cilac_t4_10_h.csv.gz",compression="gzip")
sact = sact.replace(0.0,np.nan)
sact = pd.merge(mapL,sact,on="cilac",how="outer")
if custD == "tank":
    plog("------------------------match-chirality-----------------------")
    sact.loc[:,"chi_2"] = pd.merge(sact,poi,on="id_poi",how="left")['chirality']
    sact = sact[sact["chi"] == sact["chi_2"]]
hL = sact.columns[[bool(re.search('-??T',x)) for x in sact.columns]]
hL = sorted(list(set(hL) & set(hL1)))
hL = sorted(list(set(hL) & set(hL1)))

if False:
    plt.plot(mist[hL].sum(axis=0))
    plt.show()

if custD == "mc":
    poi.loc[:,"type1"] = poi[['type','subtype']].apply(lambda x: "%s-%s" % (x[0],x[1]),axis=1)
    mist = pd.merge(mist,poi[[idField,"type1"]],on=idField,how="left")
    gist = mist.groupby("type1").agg(np.mean)
    sact = pd.merge(sact,poi[[idField,"type1"]],on=idField,how="left")

import importlib
importlib.reload(t_e)
importlib.reload(g_e)

if ops["isScore"]:
    act = t_e.joinSource(sact[[idField] + hL],mist[[idField] + hL],how="inner",idField=idField)
    scorM1 = t_e.scorPerf(act,step="etl",idField=idField)

mapN = mapL.groupby(idField).agg(len)

keepL = []
sact.loc[:,"weight"] = 0.
importlib.reload(shl)
importlib.reload(s_l)
i = np.unique(sact[idField])[6]
g = sact[sact[idField] == i]
for i,g in sact.groupby(idField):
    g = g[g[hL].sum(axis=1) > 110.]
    for i1,g1 in g.iterrows():
        g.loc[i1,hL] = s_l.interpMissing(g1[hL])
    X = g[hL].values.T
    if not X.sum().sum() > 0:
        continue
    #np.linalg.det(X)
    setL = mist[idField] == i
    if sum(setL) == 0:
        continue
    if ops['isType']:
        y = gist.loc[g['type1']].values
    else:
        y = mist[setL][hL].values[0]
    c = shl.linWeight(X,y,n_source=5)
    if False:
        X = g.loc[g.index[:10],hL]
        tlib.plotPairGrid(X.T)
    sact.loc[g.index,"weight"] = c
    if False:
        x = X.sum(axis=1)
        x1 = np.multiply(X,c).sum(axis=1)
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hL]
        pName = poi.loc[poi[idField] == i,"name"].values[0]
        #plt.title("id: %s - %s - cor %.2f " % (i,pName,sp.stats.pearsonr(x1,y)[0]))
        plt.title("id: %s - %s " % (i,pName))
        x1[x1 < 30] = 30
        #plt.plot(x/x.max(),label="raw")
        plt.plot(t,x1,label="activities")
        plt.plot(t,y,label="reference")
        plt.legend()
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.xticks(rotation=15)
        plt.show()
    
if False:
    keepL = [item for sublist in keepL for item in sublist]
    sact = sact[sact.index.isin(keepL)]
    
sact = sact.sort_values([idField,'weight']).reset_index()
print(sact[sact['weight'] > 0.].shape,sact.shape)
tact = sact.copy()
tact.loc[:,hL] = np.multiply(sact[hL].values,sact['weight'].values[:,np.newaxis])
tact = tact[[idField] + hL].groupby(idField).agg(np.sum).reset_index()
mapL.loc[:,"weight_y"] = pd.merge(mapL,sact,on="cilac",how="left")['weight']
#mapL.to_csv(baseDir + "raw/"+custD+"/map_cilac.csv",index=False)
tact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group_h.csv.gz",compression="gzip",index=False)

if False:
    importlib.reload(t_e)
    pact = t_e.joinSource(sact,mist,how="inner",idField=idField)
    pact = t_e.concatSource(pact,tact,how="inner",idField=idField,varName="weighted")
    t_e.plotSum(act)

    for i,g in pact.groupby(idField):
        plt.plot(g['ref'])
        plt.plot(g['act'])
        plt.plot(g['weighted'])
        break
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    t_e.plotSum(pact,colList=['act','ref','weighted'],ax=ax[0])
    t_e.plotSum(pact,colList=['ref','weighted'],ax=ax[1])
    plt.show()
    # gact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip",index=False)
    # pact.to_csv(baseDir + "raw/"+custD+"/act_cilac_ref.csv.gz",compression="gzip",index=False)

if ops['isScore']:
    gact = t_e.joinSource(tact,mist,how="inner",idField=idField)
    scorM2 = t_e.scorPerf(gact,step="reg",idField=idField)
    
if ops['isScore']:
    scor = pd.merge(scorM1,scorM2,on="id_poi",how="outer")
    # for i in [scorM3,scorM4]:
    #     scor = pd.merge(scor,i,on="id_poi",how="outer")
    scor.to_csv(baseDir + "raw/"+custD+"/scorMap.csv",index=False)

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



#14898
