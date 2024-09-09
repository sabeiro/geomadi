import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.train_reshape as t_r

import importlib
importlib.reload(t_r)

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

