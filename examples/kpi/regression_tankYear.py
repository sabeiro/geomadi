#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as l_c
import geomadi.series_lib as s_l
import geomadi.train_lib as tlib
import geopandas as gpd
import etl.etl_mapping as e_m
import seaborn as sns
import geomadi.train_filter as t_f

def plog(text):
    print(text)

import importlib
importlib.reload(s_l)
importlib.reload(t_l)
importlib.reload(tlib)

ops = {"etl":False,"timeSeries":False}
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi.groupby('id_clust').first().reset_index()
dateL = pd.read_csv(baseDir + "raw/tank/dateList.csv")
dateL = dateL[dateL['use'] == 1]

if ops['etl']: # etl_tankYear.py
    mist = pd.read_csv(baseDir + "raw/tank/act_vist_year.csv.gz",compression="gzip")
    mist.loc[:,"day"]  = mist['time'].apply(lambda x:x[0:10])
    mist.loc[:,"hour"] = mist['time'].apply(lambda x:x[11:13])
else: # etl_tankTimeSeries.py
    mist = pd.read_csv(baseDir + "raw/tank/act_gauss_year.csv.gz",compression="gzip")
    #mist.loc[:,"time"] = mist[['day','hour']].apply(lambda x: "%sT:%02d:00:00" %(x[0],int(x[1])),axis=1)

mist.replace(float('NaN'),0,inplace=True)
mist = mist[mist['act'] > 0]
print("total locations %d:" % (len(set(mist['id_clust']))) )

importlib.reload(tlib)
importlib.reload(l_c)

scorP, scorL = l_c.learnPlayTank(mist,dateL,poi,play=False)

vist = pd.DataFrame()
for i, (a, b) in enumerate(scorL):
    b.loc[:,"id_clust"] = a
    vist = pd.concat([vist,b],axis=0)
    
gist = vist.groupby('id_clust').apply(lambda x: np.mean(x['quot'])).reset_index()
gist.columns = ["id_clust","quot"]
gist.loc[:,"bin"], _ = t_f.binOutlier(gist["quot"],nBin=10,threshold=0.3)
poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.loc[:,"cilac_weight"] = pd.merge(poi,gist,how="left",on="id_clust")['quot']
poi.loc[:,"fact_reg"] = pd.merge(poi,scorP,how="left",on="id_clust")['cor_raw']
poi.to_csv(baseDir + "raw/tank/poi.csv",index=False)

vist = vist.drop(columns="quot")
vist.to_csv(baseDir + "raw/tank/act_vist_reg_year.csv.gz",compression="gzip",index=False)

if False:
    shl.kpiDis(scorP,"id_clust",col_cor="cor_cross",col_dif="d_20_h_l",col_sum="s_ref_h_l")
    
    tist = mist[['day','ref','act']].groupby(['day']).agg(sum).reset_index()
    plt.plot(tist['ref'],label="visit")
    plt.plot(tist['act'],label="activity")
    plt.legend()
    plt.show()
    
    tist = mist[['day','ref','act']].groupby(['day']).agg(sum).reset_index()
    tist = pd.merge(tist,dateL,on="day",how="left")
    tist.loc[:,"dif"] = (tist['act'] - tist['ref'])/(tist['act'] + tist['ref'])
    tist.loc[:,"min_temp"], _ = t_f.binOutlier(tist['Tmin'],8)
    tist.boxplot(column="dif",by="min_temp")
    xlab = ["%.0f" % x for x in _[:]]    
    plt.xticks(range(len(xlab)),xlab)
    plt.show()

    tist.boxplot(column="dif",by="wday")
    plt.show()

    correlation_matrix = tist[['wday','cloudCover','humidity','precipProbability','visibility','Tmax','holiday','dif']].corr()
    sns.heatmap(correlation_matrix, vmax=1, square=True,annot=True,cmap='RdYlGn')
    plt.show()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
