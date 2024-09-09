#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import etl.etl_mapping as e_m
import custom.lib_custom as t_l
import geomadi.series_lib as s_l
import geomadi.train_shapeLib as shl
import geomadi.train_filter as t_f
import geomadi.train_lib as tlib
import custom.lib_custom as l_c

def plog(text):
    print(text)

import importlib
importlib.reload(s_l)
importlib.reload(t_l)
importlib.reload(l_c)

ops = {"isScore":True,"lowcount":True,"p_sum":True,"isWeekday":True}

vist = pd.read_csv(baseDir + "raw/tank/ref_year.csv.gz")
vist.loc[:,"time"] = vist['time'].apply(lambda x:x[:11])
vist = vist.groupby(["id_poi","time"]).agg(np.sum).reset_index()
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.sort_values("competitor",inplace=True)
poi = poi.groupby("id_poi").first().reset_index()
vist = pd.merge(vist,poi[['id_poi','id_clust']],how='left',on="id_poi")
mist = vist.pivot_table(index=["id_clust"],columns=['time'],values="ref",aggfunc=np.sum).reset_index()

sact = pd.read_csv(baseDir + "raw/tank/act_cilac.csv.gz")
cellL = pd.read_csv(baseDir + "raw/tank/cilac_sel_tank.csv")

hL = [x for x in mist.columns if bool(re.search("T",x))]
hL1= [x for x in sact.columns if bool(re.search("T",x))]
hL = sorted(list(set(hL) & set(hL1)))

lRef = mist[hL].sum(axis=0)
lAct1 = sact[hL].sum(axis=0)

mist.replace(float('nan'),0,inplace=True)
sact.replace(float('nan'),0,inplace=True)

sact = sact[sact['id_clust'].isin(set(np.unique(mist['id_clust'])))]
sact.sort_values("id_clust",inplace=True)
mist.sort_values("id_clust",inplace=True)
sact = sact.reset_index()
mist = mist.reset_index()
cilSel = pd.DataFrame({"cilac":sact['cilac'],"weight":1.},index=sact.index)
cilSel = cilSel.iloc[sact.index]
mist = mist[['id_clust'] + hL]
sact = sact[['id_clust','cilac'] + hL]

if ops["isScore"]:
    scorM1 = l_c.tankPerf(sact,mist,step="etl")

if ops['lowcount']: #remove low counts
    selL = sact[hL].sum(axis=0)
    selL = selL > np.mean(selL)*.3
    hL = [x for x,y in zip(hL,selL.values) if y]
    mist = mist[['id_clust'] + hL]
    sact = sact[['id_clust','cilac'] + hL]
    if ops["isScore"]:
        scorM2 = l_c.tankPerf(sact,mist,step="lowcount")
    lAct2 = sact[hL].sum(axis=0)
    lRef2 = mist[hL].sum(axis=0)
  
if ops['isWeekday']:
    act = l_c.tankJoinSource(sact,mist)
    act.loc[:,"wday"] = act['time'].apply( lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT").weekday())
    act.loc[:,"dif"] = act['ref']/act['act']
    act.replace(float('inf'),1.,inplace=True)
    wday = act[['wday','dif']].groupby('wday').agg(np.mean).reset_index()
    c_wday = pd.DataFrame({"wday":[datetime.datetime.strptime(x,"%Y-%m-%dT").weekday() for x in hL]})
    c_wday = pd.merge(c_wday,wday,on="wday",how="left")
    sact.loc[:,hL] = np.multiply(sact[hL].values,c_wday['dif'][np.newaxis,:])
    lAct3 = sact[hL].sum(axis=0)

if ops["isScore"]:
    scorM3 = l_c.tankPerf(sact,mist,step="wday")
    
if ops["p_sum"]:
    #plt.plot(lAct1.values,label="act")
    #plt.plot(lRef.values,label="ref")
    plt.plot(lAct3.values,label="act wday")
    plt.plot(lRef2.values,label="ref")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()
    
act = l_c.tankJoinSource(sact,mist,how="inner")
if True:
    l_c.plotSum(act)

importlib.reload(shl)
scoreL = shl.scoreLib()
scorS = scoreL.score(sact,mist,hL,"id_clust")
scorS.loc[:,"cilac"] = cilSel['cilac']
scorS.replace(float('nan'),0,inplace=True)
plog('---------filter-good-correlating-/-multiply-by-regressor---------------')
setL = (scorS['y_cor'].abs() > 0.2)
tact = sact[setL]
if ops["isScore"]:
    scorM4 = l_c.tankPerf(tact,mist,step="f_r0.2")
    
scorS2 = scoreL.score(tact,mist,hL,"id_clust")
tact.loc[:,hL] = np.multiply(tact[hL].values,scorS2['y_reg'][:,np.newaxis])
if ops["isScore"]:
    scorM5 = l_c.tankPerf(tact,mist,step="m_reg")
tact = tact.replace(float('inf'),0.)
tact = tact.replace(float('nan'),0.)
scorS2 = scoreL.score(tact,mist,hL,"id_clust")
tact.loc[:,hL] = np.multiply(tact[hL].values,scorS2['y_reg'][:,np.newaxis])
if ops["isScore"]:
    scorM7 = l_c.tankPerf(tact,mist,step="m_reg2")

gact = tact.groupby('id_clust').sum().reset_index()
if ops["isScore"]:
    scorM6 = l_c.tankPerf(gact,mist,step="m_group")

act = l_c.tankJoinSource(gact,mist,how="inner")
act = act[act['act']>0.]
act = act[act['ref']>0.]
act = act[['id_clust','time','act','ref']]
act.to_csv(baseDir + "raw/tank/act_cilac_group.csv.gz",compression="gzip",index=False)
scorS2.to_csv(baseDir + "raw/tank/scor_set.csv",index=False)

if False:
    l_c.plotSum(act)

if ops["isScore"]:
    scorM = scorM1
    if ops['lowcount']:
        scorM = pd.merge(scorM,scorM2,on="id_clust",how="outer")
    scorM = pd.merge(scorM,scorM3,on="id_clust",how="outer")
    scorM = pd.merge(scorM,scorM4,on="id_clust",how="outer")
    scorM = pd.merge(scorM,scorM5,on="id_clust",how="outer")
    scorM = pd.merge(scorM,scorM6,on="id_clust",how="outer")
    scorM.to_csv(baseDir + "raw/tank/scor_cilacYear.csv",index=False)
    
if False:
    lL = ["all","filtered","group","group_selection"]
    for i,g in enumerate([scorS,scorT,scorG,scorP]):
        y = g['y_cor'].replace(float('nan'),0)
        plt.hist(y,bins=20,normed=1,histtype='step',cumulative=-1,label=lL[i],linewidth=1.5,)
    plt.legend()
    plt.show()
    
if False:
    scorS[['y_cor','y_reg','y_ent']].plot.kde(bw_method=0.3)
    plt.show()
    tL = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in hL]
    plt.plot(gact[hL].sum(axis=0),label="activities")
    plt.plot(mist[hL].sum(axis=0),label="visits")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

    

