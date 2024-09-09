import os, sys, gzip, random, csv, json, datetime, re, time
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_model as tlib
import geomadi.train_modelList as modL
import geomadi.train_shape as shl
import geomadi.train_execute as t_e
import geomadi.train_metric as t_m
import geomadi.train_score as t_s
import importlib
from io import StringIO
from sklearn import decomposition
from sklearn.decomposition import FastICA, PCA
import pymongo
import osmnx as ox
import shapely as sh

print('-------------------load/def------------------------')
ops = {"isScore":True,"lowcount":True,"p_sum":True,"isWeekday":True,"isType":False}
fSux = "20"
idField = "id_poi"
custD = "tank"
projDir = baseDir + "raw/"+custD+"/act_cilac_11/"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

fL = os.listdir(projDir)
fL = [x for x in fL if bool(re.search("t4_",x))]
f = fL[0]
fName = projDir + f

mapL  = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv.gz",compression="gzip")
mapL.loc[:,"weight"] = 0.
mapL = mapL.groupby([idField,"cilac"]).first().reset_index()
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL = dateL[dateL['use'] == 3]
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
mist.loc[:,idField] = mist[idField].astype(str)
hL1 = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]
scor = []
scorD = []
f = fL[0]
fName = projDir+f
sact, hL = t_e.loadDf(fName,dateL,poi,mapL,custD)
hL1 = t_r.timeCol(mist)
hL = t_r.overlap(hL,hL1)
hL = [x for x in hL if x > '2019-01-06T']


import importlib
importlib.reload(shl)
importlib.reload(t_e)
importlib.reload(t_m)
importlib.reload(t_s)
importlib.reload(modL)
importlib.reload(tlib)
importlib.reload(shl)

if True:
    print('---------------------------fit------------------------')
    importlib.reload(t_r)
    XL, yL, idL = t_r.id2group(sact,mist,hL,idField)
    tml = modL.modelList()
    weiL = []
    pact = []
    i = '1218'
    for j,i in enumerate(idL):
        g = sact[sact[idField] == i]
        X, y = XL[j], yL[j]
        clf = tml.regL['elastic_cv']['mod']
        fit_w = clf.fit(X,y)
        w = fit_w.coef_
        y1 = fit_w.predict(X)
        y2 = np.multiply(X.T,w[:,np.newaxis]).sum(axis=0)
        r = np.apply_along_axis(lambda x: sp.stats.pearsonr(x,y)[0],axis=1,arr=X.T)
        weiL.append(pd.DataFrame({idField:i,"cilac":g['cilac'],"weight":w,"cor":r}))
        pact.append(pd.DataFrame({idField:i,"act":y1,"ref":y,"time":hL}))
        
    weiL = pd.concat(weiL)
    pact = pd.concat(pact)

if False:
    print('------------------------plot-last-line--------------------')
    plt.title("id_poi %s cor %.2f" % (i,sp.stats.pearsonr(y,y1)[0]))
    plt.plot(y1,label="pred")
    plt.plot(y,label="ref")
    plt.plot(y2,label="mult")
    plt.show()
    
print('---------------------------------collect--------------------------')
tact = sact.copy()
tact.loc[:,"weight"] = pd.merge(tact,weiL,on=[idField,"cilac"],how="left")['weight_y']
tact.loc[:,hL] = np.multiply(tact[hL].values,tact['weight'].values[:,np.newaxis])
tact = tact[[idField] + hL].groupby(idField).agg(np.sum).reset_index()
mapL = mapL.sort_values([idField,'cilac'])
mapL = pd.merge(mapL,weiL,on=[idField,"cilac"],how="left",suffixes=["_x",""])
del mapL['weight_x']
mapL.to_csv(baseDir + "raw/"+custD+"/map_cilac.csv.gz",compression="gzip",index=False)
weiL.to_csv(baseDir + "raw/"+custD+"/map_february.csv.gz",compression="gzip",index=False)
tact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group_19.csv.gz",compression="gzip",index=False)
print("unique poi %d " % len(set(tact[idField])))

if True:
    i = '1001'
    p1 = tact[tact[idField] == i]
    p2 = pact[pact[idField] == i]
    p3 = mist.loc[mist[idField] == i]
    p3 = p3[p3.columns[p3.columns >= '2019-01-01']]
    for y in [p1,p2,p3]:
        del y[idField]
    t1 = t_r.day2time(t_r.timeCol(p1))
    t2 = t_r.day2time(p2['time'])
    t3 = t_r.day2time(t_r.timeCol(p3))
    plt.title("id_poi %s " % (i) )
    plt.plot(t1,p1.values[0],label="extension")
    plt.plot(t3,p3.values[0],label="reference")
    plt.plot(t2,p2['act'],label="pact")
    plt.legend()
    plt.xticks(rotation=15)
    plt.show()
# s = pd.merge(t1,t2,on=[idField,"cilac"],how="left")
# s.to_csv(baseDir + "tmp/merge.csv")
# weiL[weiL[idField].isin(poi.loc[poi['use']==3,idField])].to_csv(baseDir + "tmp/map_cilac.csv")

if ops['isScore']:
    print('-----------------correlating-cells------------------')
    importlib.reload(t_s)
    act = t_e.joinSource(sact,mist,how="inner",idField=idField)
    scorM1 = t_s.scorPerf(act,step="raw",idField=idField)
    act = t_e.joinSource(tact,mist,how="inner",idField=idField)
    scorM2 = t_s.scorPerf(act,step="map",idField=idField)
    scor.append(scorM1)
    scor.append(scorM2)
    scorD.append({"idx":lab,"raw":np.mean(scorM1["r_raw"]),"map":np.mean(scorM2["r_map"])})
    scorD = pd.DataFrame(scorD)
    scor = pd.concat(scor,axis=1)
    print(scorD)
    scor = pd.merge(scorM1,scorM2,on="id_poi",how="outer")
    if False:
        print('-------------------add-performace-of-best-cilac----------------------')
        cilacScor, missingL = t_s.score(sact,mist,hL,idField=idField,idFeat="cilac")
        cilacScor.sort_values("s_cor",ascending=False,inplace=True)
        scorM3 = cilacScor.groupby(idField).first().reset_index()
        for i in [scorM3]:
            scor = pd.merge(scor,i,on="id_poi",how="outer")
    scor.to_csv(baseDir + "raw/"+custD+"/scor/scorMap.csv",index=False)

    
if False:
    print('----------------------checking-one-location----------------------')
    i = '1413'
    t = sact[sact[idField] == i]
    m = mist[mist[idField] == i]
    X = t[hL].values
    y = m[hL].values[0]
    r = np.apply_along_axis(lambda x: sp.stats.pearsonr(x, y),axis=1,arr=X)[:,0]
    d = pd.DataFrame({"cilac":t['cilac'],"cor":r})
    d.to_csv(baseDir+"tmp/ciccia.csv",index=False)
    plt.plot(X.sum(axis=0),linewidth=2)
    plt.plot(y,linewidth=4)
    for i in range(X.shape[0]):
        plt.plot(X[i])
    plt.show()

#scorM2[['id_poi','r_map_t4_p11_d40']]

    
if False:
    i = 1
    g = sact[sact[idField] == poi.iloc[i][idField]]
    X = g[hL].values.T #np.linalg.det(X)
    setL = mist[idField] == poi.iloc[i][idField]
    y = mist[setL][hL].values[0]

if True:
    gact = pd.read_csv(baseDir + "raw/"+custD+"/act_cilac_group_19.csv.gz",compression="gzip")
    gact[idField] = gact.astype(str)
    for i in ["/act_cilac_group_18.csv.gz"]:
        g = pd.read_csv(baseDir + "raw/"+custD+i,compression="gzip")
        g[idField] = g.astype(str)
        g = g[g[idField].isin(poi.loc[poi['use']==3,idField])]
        gact = pd.merge(gact,g,on=idField,how="left",suffixes=["","_y"])
    gact = gact[[x for x in gact.columns if not bool(re.search("_y",x))]]
    hL1 = sorted([x for x in gact.columns if bool(re.search("T",x))])
    gact = gact[[idField]+hL1]
    gact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip",index=False)

if False:
    print('-------------------export-cilacs-for-tibco-------------------')
    mact = pd.melt(sact,id_vars=["cilac",idField,"chi"],value_vars=hL)
    mact.columns = ["cilac",idField,"chi","day","count"]
    mact.to_csv(baseDir+"gis/"+custD+"/act_cilac.csv",index=False)
    mact = pd.melt(mist,id_vars=idField,value_vars=hL)
    mact.columns = [idField,"day","count"]
    mact.to_csv(baseDir+"gis/"+custD+"/ref.csv",index=False)
    
if False:
    keepL = [item for sublist in keepL for item in sublist]
    sact = sact[sact.index.isin(keepL)]
    
if False:
    plt.title("percentage of location correlation over 0.6")
    plt.bar(scorD['idx'],height=scorD['map'])
    plt.xticks(rotation=15)
    plt.show()
    rL = [x for x in scor.columns if bool(re.search("r_",x))]
    scor.boxplot(column=rL)
    plt.xticks(rotation=15)
    plt.ylabel("correlation")
    plt.show()

if False:
    importlib.reload(t_e)
    pact = t_e.joinSource(sact,mist,how="inner",idField=idField)
    pact = t_e.concatSource(pact,tact,how="inner",idField=idField,varName="weighted")
    t_e.plotSum(act)
    fig, ax = plt.subplots(1,2)
    t_e.plotSum(pact,colList=['act','ref','weighted'],ax=ax[0])
    t_e.plotSum(pact,colList=['ref','weighted'],ax=ax[1])
    plt.show()
    # gact.to_csv(baseDir + "raw/"+custD+"/act_cilac_group.csv.gz",compression="gzip",index=False)
    # pact.to_csv(baseDir + "raw/"+custD+"/act_cilac_ref.csv.gz",compression="gzip",index=False)

if False:
    print('----------------plot-filter-sums---------------')
    sumD = []
    for f in fL:
        lab = f.split(".")[0]
        lab = "_".join(lab.split("_")[2:])
        sact, hL = t_e.loadDf(projDir+f,dateL,poi,mapL,custD,hL1=hL1)
        tL = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in hL]
        y = sact[hL].sum(axis=0)
        sumD.append(pd.DataFrame({lab:y}))
        plt.plot(tL,y,label=lab)
    norm = y.sum()
    y = mist.loc[mist[idField].isin(poi[idField]),hL].sum(axis=0)
#    y = y*norm/y.sum()
    plt.plot(tL,y,label="reference",linewidth=3)
    plt.legend()
    plt.xticks(rotation=15)
    plt.show()
    sumD = pd.concat(sumD,axis=1)
    sumD.boxplot()
    plt.ylabel("counts")
    plt.show()

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



