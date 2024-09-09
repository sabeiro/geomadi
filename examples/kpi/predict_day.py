import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.series_stat as s_s
import geomadi.train_modelList as t_m
import geomadi.train_model as tlib

print('-----------------------------------define-----------------------')
idField = "id_poi"
custD = "tank"
version = "11u"
#version = "prod"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL = dateL[dateL['day'] > '2018-12-31T']
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: str(x)+"T")
if False:
    poi = poi[poi['use'] == 3]
    dateL = dateL[dateL['day'] >= '2019-01-01']

print('-------------------------load-join-clean---------------------------------')
tact = pd.read_csv(baseDir+"raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",dtype={idField:str})
mist = pd.read_csv(baseDir+"raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip",dtype={idField:str})
dirc = pd.read_csv(baseDir+"raw/"+custD+"/dirCount/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
#dirc = t_r.mergeDir(baseDir+"raw/"+custD+"/viaCount/")
ical = pd.read_csv(baseDir+"raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical = t_r.isocal2day(ical,dateL)
tist = t_e.joinSource(tact,mist,how="left",idField=idField,isSameTime=True)
#tist = tist[tist[idField].isin(poi[idField])]
#tist = tist[tist.columns[tist.columns >= '2018-09-00T']]
tist = t_e.concatSource(tist,ical,how="left",idField=idField,varName="ical")
tist = t_e.concatSource(tist,dirc,how="left",idField=idField,varName="foot")
print("total days %d" % (len(set(tist['time']))))
norm = 0.4226623389831198
tist.loc[:,"foot"] = tist["foot"]*norm*1.1
dateL.loc[:,"time"] = dateL['day']
tL = ['time','wday','cloudCover','humidity','visibility','Tmax']
tist = tist.merge(dateL[tL],on="time",how="left")
tist = tist[tist['time'] == tist['time']]
tist.replace(np.nan,0,inplace=True)
hL = np.unique(tist['time'])
testL, trainL, validL = [x < '2019-02-01T' for x in hL], [(x >= '2019-02-01T') & (x < '2019-03-01T') for x in hL], [(x >= '2019-03-01T') for x in hL]
print("sample train %d test %d valid %d" % (sum(testL),sum(trainL),sum(validL)) )

sL = ['act','ical','foot','wday','cloudCover','humidity','visibility','Tmax']
setL = tist['ref'] != tist['ref']
tist.loc[setL,"ref"] = tist.loc[setL,'ical']
tist.loc[testL,"foot"] = s_s.interpMissing(tist.loc[testL,'foot'])
if False:
    for i,g in tist.groupby(idField):
        if i == '1014':
            break
    t_v.plotTimeSeries(g[sL+['ref']],hL=g['time'])
    plt.show()

XL, yL = t_r.id2dim(tist,sL,idField)

import importlib
importlib.reload(t_e)
importlib.reload(t_m)
importlib.reload(t_r)
importlib.reload(tlib)

tMod = tlib.regName(XL,yL,modName="perceptron")
predL, fitL, scorV = tMod.tuneReg(trainL,testL,nIter=40,paramF=baseDir+"train/reg_"+custD+".json")
#predL, fitL, scorS = tMod.runReg(trainL,validL,paramF=baseDir+"train/reg_"+custD+".json")
scorV = t_v.plotHyperPerf(scorV)

scorV.to_csv(baseDir + "raw/"+custD+"/scor/scor_feature.csv",index=False)
#t_v.plotFeatCorr(scorM)

clf = tMod.getModel()

tist.loc[:,"pred"] = 0.
scorM = []
for i,g in tist.groupby(idField):
    X = g[sL].values
    x = X[:,0]
    y = g['ref'].values
    r1 = sp.stats.pearsonr(X[:,0][trainL],y[trainL])[0]
    r2 = sp.stats.pearsonr(X[:,0][testL],y[testL])[0]
    r3 = sp.stats.pearsonr(X[:,0][validL],X[:,1][validL])[0]
    fit_w = clf.fit(X[trainL],y[trainL])
    scor = {}
    y_pred = fit_w.predict(X[trainL])
    scor["cor_train"] = sp.stats.pearsonr(y[trainL],y_pred)[0]
    scor["cor_input"] = sp.stats.pearsonr(x[trainL],y_pred)[0]
    y_pred = fit_w.predict(X[testL])
    scor["cor_test"] = sp.stats.pearsonr(y[testL],y_pred)[0]
    y = fit_w.predict(X)
    tist.loc[g.index,"pred"] = y
    if False:
        plt.title("id_poi %s cor input %.2f cor pred %.2f" % (i,scor['cor_input'],scor['cor_test']))
        plt.plot(y_pred,label="pred")
        plt.plot(y[testL],label="ref test")
        plt.plot(X[testL][:,0],label="act")
        plt.legend()
        plt.show()
    if False:
        y_2 = (g['act'].values + y)*.5
        plt.title("id_poi %s cor train %.2f cor test %.2f cor iso %.2f" % (i,r1,r2,r3))
        plt.plot(g['act'],label="act")
        plt.plot(g['ref'],label="ref",linewidth=3)
        plt.plot(g['ical'],label="isocal")
        plt.plot(y,label="pred")
        plt.plot(y_2,label="av")
        plt.legend()
        plt.show()

if False:
    i = 0
    X_valid = XL[i][:,:][validL]
    X_test  = XL[i][:,:][testL]
    X_train = XL[i][:,:][trainL]
    x_test = XL[i][:,0][testL]
    y_train = yL[i][trainL]
    y_test = yL[i][testL]
    fit_w = clf.fit(X_train,y_train)
    y_pred = fit_w.predict(X_test)
    #y_pred = fit_w.predict(X_valid)
    r1 = sp.stats.pearsonr(y_test,y_pred)[0]
    r2 = sp.stats.pearsonr(y_test,x_test)[0]
    plt.title("id_poi %d cor input %.2f cor pred %.2f" % (i,r1,r2))
    plt.plot(y_test,label="ref test")
    plt.plot(y_pred,label="pred")
    plt.plot(x_test,label="act")
    plt.legend()
    plt.show()
    i = 0
    y = yL[i][validL]
    p = predL[i]
    plt.plot(y,label="reference")
    plt.plot(x,label="input")
    plt.plot(p,label="prediction")
    plt.legend()
    plt.show()

    x_valid = XL[i][:,:][validL]
    x_train = XL[i][:,:][trainL]
    fig, ax = plt.subplots(1,1)
    t_v.plotTimeSeries(pd.DataFrame(x_train),ax=ax)
    t_v.plotTimeSeries(pd.DataFrame(x_valid),ax=ax)
    plt.show()

    


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
