#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
#import modin.pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import etl.etl_mapping as e_m
import geomadi.series_lib as s_l
import custom.lib_custom as l_c
import geomadi.train_lib as tlib

def plog(text):
    print(text)

ops = {"twoStage":True}
    
dateL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi.groupby('id_clust').first().reset_index()

if True:
    plog('------------------radom days out-----------------------')
    dateL.loc[:,"use"] = 1
    shuffleL = random.sample(range(dateL.shape[0]),dateL.shape[0])
    dateL.loc[dateL.index[shuffleL][:30],"use"] = 2

if ops['twoStage']:
    tist = pd.read_csv(baseDir + "raw/tank/act_gauss_year.csv.gz",compression="gzip")
else:
    tist = pd.read_csv(baseDir + "raw/tank/act_cilac_group.csv.gz",compression="gzip")
    tist.loc[:,"day"] = tist['time'].apply(lambda x:x[:10])

if False:
    plog('-------------check model autocompatibility-------------')
    tist.loc[:,"act"] = tist['ref']*(1.+np.random.randn(tist.shape[0])*.2)

tist = tist[tist["day"].isin(dateL["day"])]
if False: 
    plog('------------small sample------------------------')
    tist = tist[tist["id_clust"].isin(np.unique(tist['id_clust'])[:3])]

if False:
    l_c.plotSum(tist,isLoc=False)

scorM = l_c.tankPerf1(tist,step="interp")

import importlib
importlib.reload(tlib)
importlib.reload(s_l)
importlib.reload(l_c)

if ops['twoStage']:
    scorLearn, scorL = l_c.learnPlayTank(tist,dateL[dateL['use']==1],poi,play=False)
    scorLearn.to_csv(baseDir + "raw/tank/scor_learn.csv",index=False)
    scorPlay, scorL  = l_c.learnPlayTank(tist,dateL,poi,play=True)
    scorPlay.to_csv( baseDir + "raw/tank/scor_play.csv",index=False)
    vist = pd.DataFrame()
    for i, (a, b) in enumerate(scorL):
        b.loc[:,"id_clust"] = a
        vist = pd.concat([vist,b],axis=0)
    vist = vist.drop(columns="quot")
else:
    scorLearn, vist = l_c.learnPlayTankDay(tist,dateL[dateL['use']==1],poi,play=False)
    scorLearn.to_csv(baseDir + "raw/tank/scor_learnDay.csv",index=False)
    scorPlay , vist  = l_c.learnPlayTankDay(tist,dateL,poi,play=True)
    setL = vist['day'].isin(dateL[dateL['use']==2]['day'])
    mist = vist.copy()
    scorB = l_c.tankPerf1(mist[setL],step="blind")
    mist.loc[:,"act"] = l_c.smoothClust(mist['act'],mist['id_clust'],width=1,steps=3)
    mist.loc[:,"ref"] = l_c.smoothClust(mist['ref'],mist['id_clust'],width=1,steps=3)
    scorS = l_c.tankPerf1(mist[setL],step="smooth")
    scorP = pd.merge(scorPlay,scorB,on="id_clust",how="outer")
    scorP = pd.merge(scorP,scorS,on="id_clust",how="outer")
    scorP.to_csv( baseDir + "raw/tank/scor_playDay.csv",index=False)

if False:
#    del mist['time']
    corL = []
    setL = mist['day'].isin(dateL[dateL['use']==2]['day'])
    for i,g in mist[setL].groupby(["day"]):
        corL.append({"day":g['day'].iloc[0],"cor":sp.stats.pearsonr(g['act'],g['ref'])[0]})
    corL = pd.DataFrame(corL)
    plt.plot(corL['day'],corL['cor'])
    plt.xlabel("day")
    plt.ylabel("correlation")
    plt.xticks(rotation=25)
    plt.show()
    
if False:
    l_c.plotSum(mist[setL],isLoc=False)
    l_c.plotSum(mist[setL],isLoc=True)
    l_c.plotSum(vist,isLoc=False)

if False:
    shl.kpiDis(scorPlay,"id_clust",col_cor="cor_cross",col_dif="diff",col_sum="sum")
    D = scorL[0][1]
    Dd  = D[ ['day','ref','act']].groupby(['day']).agg(sum).reset_index()

    plt.plot(Dd['act'])
    plt.plot(Dd['ref'])
    plt.show()

    plt.plot(X1['value'].values)
    plt.plot(X2['value'].values)
    plt.plot(X1['corr'].values)
    plt.show()
    
vist.to_csv(baseDir + "raw/tank/act_vist_play.csv.gz",compression="gzip",index=False)

