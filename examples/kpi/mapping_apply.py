import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_reshape as t_r
import geomadi.train_execute as t_e

import importlib
importlib.reload(t_r)

idField = "id_poi"
custD = "tank"
version = "prod"
projDir = baseDir + "raw/"+custD+"/act_cilac_"+version+"/"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
mapL  = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv.gz",compression="gzip")
weiL  = pd.read_csv(baseDir + "raw/"+custD+"/mapping/map_v"+version+".csv.gz",compression="gzip")
weiL.loc[:,idField] = weiL[idField].astype(str)
#mapL = mapL.dropna()
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL = dateL[dateL['day'] >= '2019-01-01']
fL = os.listdir(projDir)
fL = sorted([x for x in fL if bool(re.search("csv",x))])
lact = []
for f in fL:
    fName = projDir+f
    sact, hL = t_e.loadDf(fName,dateL,poi,mapL,isChi=False,hL1=[None])
    X = sact[hL].values
    print(X.sum().sum()/len(hL))
    tact = sact.copy()
    tact.loc[:,"weight"] = tact.merge(weiL,on=[idField,"cilac"],how="left")['weight_y']
    tact.loc[:,hL] = np.multiply(tact[hL].values,tact['weight'].values[:,np.newaxis])
    hL = [x for x in hL if x != '2019-04-19T']
    tact = tact[[idField] + list(hL)].groupby(idField).agg(np.sum).reset_index()
    t = sact[sact[idField] == '1001']
    print(t.shape)
    #tact = tact[tact.columns[:-1]]
    lact.append(tact)
if len(fL) > 1:
    tact = pd.merge(lact[0],lact[1],on=idField,how="outer")
if len(fL) > 2:
    tact = tact.merge(lact[2],on=idField,how="outer")
tact.to_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",index=False)

    

if True:
    print('-----------------------check-map-application---------------------')
    pact = pd.read_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip")
    deli = pd.read_csv(baseDir + "raw/"+custD+"/delivery/act_ref_foot_19_03.csv")
    mist = mist[mist.columns[mist.columns >= '2019-01-01T']]
    mist[idField] = mist[idField].apply(lambda x: str(int(float(x))))
    ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip")
    dL = t_r.day2time([x + "T" for x in dateL['day']])
    iL = ["%02d-%02dT" % (x.isocalendar()[1],x.isocalendar()[2]) for x in dL]
    ical = ical[ical.columns[ical.columns.isin( [idField] + list(set(dateL['isocal'])) )  ]]
    ical = ical.rename(columns={x:y+"T" for x,y in zip(dateL.loc[:,'isocal'],dateL.loc[:,'day'])})
    ical[idField] = ical[idField].apply(lambda x: str(int(float(x))))
    i = '1031'
    i = '1038'
    i = '1014'
    i = '1001'
    i = '1413'
    for i in set(poi.loc[poi['use'] == 3,idField]):
        y1 = pact.loc[pact[idField] == i]
        y2 = tact.loc[tact[idField] == i]
        y3 = mist.loc[mist[idField] == i]
        y3 = y3[y3.columns[y3.columns >= '2019-01-01']]
        y4 = ical.loc[ical[idField] == i]
        for y in [y1,y2,y3,y4]:
            del y[idField]
        t1 = t_r.day2time(t_r.timeCol(y1))
        t2 = t_r.day2time(t_r.timeCol(y2))
        t3 = t_r.day2time(t_r.timeCol(y3))
        t4 = t_r.day2time(t_r.timeCol(y4))
        setL = y1.columns[y1.columns.isin(y3.columns)]
        setL1 = [x for x in setL if x[5:7] == '02']
        r2 = sp.stats.pearsonr(y1[setL1].values[0],y3[setL1].values[0])[0]
        setL = y2.columns[y2.columns.isin(y4.columns)]
        setL1 = [x for x in setL if x[5:7] == '04']
        r3 = sp.stats.pearsonr(y2[setL1].values[0],y4[setL1].values[0])[0]
        plt.figure(figsize=(8,4))
        plt.title("id_poi %s cor: feb %.2f april %.2f" % (i,r2,r3) )
        plt.plot(t2,y2.values[0],label="extension")
        plt.plot(t3,y3.values[0],label="reference",linewidth=3)
        plt.plot(t1,y1.values[0],'-.',label="weighting")
        plt.plot(t4,y4.values[0],label="isocalendar")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()
        break

if False:
    print('--------------------------prepare-capture-rate------------------------')
    poi = poi[poi['use'] == 3]
    tist = t_e.joinSource(tact[tact[idField].isin(poi[idField])],mist,how="inner",idField=idField)
    tist = t_e.concatSource(tist,ical,how="left",idField=idField,varName="ical")
    tist = t_e.concatSource(tist,dirc,how="left",idField=idField,varName="foot")
    norm = 0.4226623389831198
    tist.loc[:,"foot"] = tist["foot"]*norm*1.1
    tist.loc[:,"capture_rate"] = tist["ref"]/tist["foot"]*100
    print(tist.describe())
    tist.to_csv(baseDir + "tmp/capture_rate.csv",index=False)

    deli = pd.read_csv(baseDir + "raw/" + custD + "/delivery/" + "act_ref_foot_19_02.csv")
    hL = t_r.timeCol(dirc)
    tirc = pd.melt(dirc,id_vars=idField,value_vars=hL)
    tirc = tirc[tirc[idField].isin(poi[idField])]
    tirc = tirc[tirc['variable'] > '2019-02-28T']
    tirc.loc[:,'value'] = tirc['value']*.5
    tirc.to_csv(baseDir + "tmp/direction_count.csv",index=False)
    fig, ax = plt.subplots(1,1)
    bx1 = tirc.boxplot(by=idField,column="value",ax=ax,return_type="dict")
    bx2 = deli.boxplot(by=idField,column="direction_count",ax=ax,return_type="dict")
    [[item.set_color('g') for item in bx1[key]['boxes']] for key in bx1.keys()]
    [[item.set_color('r') for item in bx2[key]['boxes']] for key in bx2.keys()]
    plt.show()



if False:
    XL, yL = t_r.id2dim(tist,sL,idField)
    tMod = tlib.regName(XL,yL,modName="perceptron")
    #predL, scorS = tMod.tuneReg(trainL,validL,nIter=40,paramF=baseDir+"train/reg_"+custD+".json")
    predL, scorS = tMod.runReg(trainL,validL,paramF=baseDir+"train/reg_"+custD+".json")
    plog('---------------plot-prediction-vs-reference-----------------')
    i = 1
    for i in range(XL.shape[0]):
        y = yL[i][validL]
        y1 = predL[i]
        err = 2.*np.sqrt( ((y-y1)**2).sum() )/(y+y1).sum()
        cor = sp.stats.pearsonr(y,y1)[0]
        plt.title("location %d err %.3f corr %.2f" % (i,err,cor))
        plt.plot(y1,label="act")
        plt.plot(y,label="ref")
        plt.legend()
        plt.show()


