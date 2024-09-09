#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as t_l
import geomadi.series_lib as s_l
import etl.etl_mapping as e_m
import geomadi.train_filter as t_f
import geomadi.train_shapeLib as t_s
import custom.lib_custom as l_c

import importlib
importlib.reload(s_l)
importlib.reload(t_l)
importlib.reload(t_s)
importlib.reload(l_c)

def plog(text):
    print(text)

tist = pd.read_csv(baseDir + "raw/tank/act_vist_year.csv.gz",compression="gzip")
tist = tist[tist['act'] > 0]
tist.loc[:,"day"]  = tist['time'].apply(lambda x:x[0:10])
tist.loc[:,"hour"] = tist['time'].apply(lambda x:x[11:13])
tist.sort_values(['id_clust','time'],inplace=True)
gist = tist[['hour','act','ref']].groupby('hour').agg(np.mean).reset_index()
XL = {}
importlib.reload(t_s)
gl = []
chiSq = {}
for i in ["ref","act"]:
    sact = tist.pivot_table(index=["id_clust","day"],columns="hour",values=i,aggfunc=np.sum).reset_index()
    #sact.loc[:,"wday"] = sact['day'].apply(lambda x:datetime.datetime.strptime(x,"%Y-%m-%d").weekday())
    hL = [x for x in sact.columns.values if bool(re.match("^[-+]?[0-9]+$",str(x)))]
    X = sact[hL].values#[(10*7):(11*7)]
    redF = t_s.reduceFeature(X)
    redF.interpMissing()
    gl.append(redF.getMatrix()[:7])
    #redF.fit(how="poly")
    gl.append(redF.getMatrix()[:7])
    redF.smooth(width=3,steps=7)
    gl.append(redF.getMatrix()[:7])
    dayN = redF.replaceOffChi(sact['id_clust'],threshold=0.03,dayL=sact['day'])
    dayN[dayN['count']>30].to_csv(baseDir + "raw/tank/poi_anomaly_"+i+".csv",index=False)        
    XL[i] = redF.getMatrix()   
    
if False:
    plt.figure()
    plt.title("image representation of time series")
    j = [1,5,9,3,7,11]
    for i in range(6):
        ax = plt.subplot(3,4,j[i])
        plt.plot(gl[i].ravel())
        ax = plt.subplot(3,4,j[i]+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(gl[i])

    plt.xlabel("hours")
    plt.ylabel("frequency")
    plt.show()
    
print("overall correlation %.3f" % sp.stats.pearsonr(tist['ref'],tist['act'])[0])
X1 = XL['act']
X2 = XL['ref']
print("correlation after smoothing %.3f" % sp.stats.pearsonr(X1.ravel(),X2.ravel())[0])
if True:
    X1 = l_c.weekdayFact(X1,X2,sact['day'])
    print("correlation after weekday correction %.3f" % sp.stats.pearsonr(X1.ravel(),X2.ravel())[0])
    y1 = X1.sum(axis=1)
    y2 = X2.sum(axis=1)
    print("daily correlation %.3f" % sp.stats.pearsonr(y1,y2)[0])

if False:
    plt.plot(X1[tL].values.ravel(),label="act",alpha=0.3)
    plt.plot(X2[tL].values.ravel(),label="ref",alpha=0.3)
    plt.legend()
    plt.show()

X1 = pd.DataFrame(X1)
X1.loc[:,"day"] = sact['day']
X1.loc[:,"id_clust"] = sact['id_clust']
X2 = pd.DataFrame(X2)
X2.loc[:,"day"] = sact['day']
X2.loc[:,"id_clust"] = sact['id_clust']

X1.replace(float('nan'),0,inplace=True)
tL = X1.columns[:-2]
tL = [int(x) for x in tL]
vist = pd.melt(X1,id_vars=["id_clust","day"],value_vars=tL,var_name="hour",value_name="act")
vist.loc[:,"ref"] = pd.melt(X2,id_vars=["id_clust","day"],value_vars=tL,var_name="hour",value_name="ref")['ref']

def clampF(x):
    return pd.Series({"r_interp":sp.stats.pearsonr(x['act'],x['ref'])[0]})
scorM = vist.groupby(["id_clust"]).apply(clampF).reset_index()
scorM.to_csv(baseDir + "raw/tank/scor_interp.csv",index=False)
print("score %.2f" % (scorM[scorM['r_interp'] > 0.6].shape[0]/scorM.shape[0]) )
vist.to_csv(baseDir + "raw/tank/act_gauss_year.csv.gz",compression="gzip",index=False)

if False:
    cist1 = vist[['id_clust','day','ref']].groupby(['id_clust','day']).agg(sum).reset_index()
    cist2 = tist[['id_clust','day','ref']].groupby(['id_clust','day']).agg(sum).reset_index()
    cist = pd.merge(cist1,cist2,on=["id_clust","day"],how="inner")
    cist.loc[:,"dif_ref"] = abs((cist['ref_x'] - cist['ref_y'])/(cist['ref_x'] + cist['ref_y']))
    print("smoothing difference: %.2f" % (100*cist['dif_ref'].mean()))

if False:
    gist = vist[['day','hour','ref','act']].groupby(['day','hour']).agg(sum)
    plt.plot(gist['ref'].values)
    plt.plot(gist['act'].values)
    plt.show()
    print(sp.stats.pearsonr(gist['ref'].values,gist['act'].values)[0])
    print(sp.stats.pearsonr(vist['ref'].values,vist['act'].values)[0])
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
