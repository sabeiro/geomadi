#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

idField = "id_poi"
custD = "tank"
poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = poi[poi['use'] == 3]
mapL  = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
fL = os.listdir(baseDir + "raw/"+custD+"/act_cilac/")
fL = [x for x in fL if bool(re.search("csv",x))]
lact = []
for f in fL:
    fName = baseDir + "raw/"+custD+"/act_cilac/"+f
    sact, hL = t_e.loadDf(fName,dateL,poi,mapL,custD,hL1=[None])
    tact = sact.copy()
    tact.loc[:,hL] = np.multiply(sact[hL].values,sact['weight'].values[:,np.newaxis])
    tact = tact[[idField] + list(hL)].groupby(idField).agg(np.sum).reset_index()
    lact.append(tact)

tact = pd.merge(lact[0],lact[i],on=idField,how="outer")
tact.to_csv(baseDir + "raw/"+custD+"/act_weighted.csv.gz",compression="gzip",index=False)

mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
#tact[idField] = tact[idField].apply(lambda x: str(int(float(x))))
mist[idField] = mist[idField].apply(lambda x: str(int(float(x))))
act = t_e.joinSource(sact,mist,how="inner",idField=idField)
scorM1 = t_s.scorPerf(act,step="raw_"+lab,idField=idField)
act = t_e.joinSource(tact,mist,how="inner",idField=idField)
scorM2 = t_s.scorPerf(act,step="map_"+lab,idField=idField)

if False:
    iL = np.unique(act[idField])
    for i,g in act.groupby(idField):
        plt.plot(g['act'])
        plt.plot(g['ref'])
        plt.show()

if False:
    XL, yL = tlib.id2dim(tist,sL,idField)
    tMod = tlib.regName(XL,yL,modName="perceptron")
    predL, scorS = tMod.tuneReg(trainL,validL,nIter=40,paramF=baseDir+"train/reg_"+custD+".json")
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

def getMarketShare(df):
    mshare = pd.read_csv(baseDir+"raw/cilacMarketshare.csv.tar.gz",compression='gzip',sep=',',quotechar='"',names=["cilac","factor"],header=0)
    factL = pd.merge(df,mshare,on="cilac",how="left")["factor"].values
    df.loc[:,"market_share"] = 1./factL
    df = df.replace(np.nan,np.nanmean(df['market_share']))
    return df

def df2Map(df):
    mapL = {}
    for i,x in df.iterrows():
        mapL[x['cilac']] = {x['zone']:{"market_share":x['market_share'],"weight":x['weight']}}
    return {"mapping":mapL}

def map2df(mapD):
    mapL = mapD['mapping']
    mapDf = []
    for i,x in mapL.items():
        for j,y in x.items():
            mapDf.append([i,j,y['weight'],y['market_share']])
    mapDf = pd.DataFrame(mapDf,columns=["cilac","zone","weight","market_share"])
    return mapDf

def writeJson(mapD,fName):
    with open(fName,"w") as f:
        f.write(json.dumps(mapD))

def exportMap(df,fName):
    writeJson(df2Map(df),fName)

def df2String(df):
    strJs = '{"mapping":'
    for j,cl in df.iterrows():
        strJs += '{"'+cl['cilac']+'":{"'+str(cl['zone'])+'":{"market_share":'+str(cl['market_share'])+',"weight":'+str(cl['weight'])+'}}},'
    strJs = strJs[:-1]
    strJs += "}"
    return strJs

