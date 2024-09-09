import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import geomadi.series_stat as s_l
import geomadi.train_execute as t_e
import geomadi.train_score as t_s
import geomadi.train_model as tlib
import geomadi.train_shape as shl
import geomadi.train_reshape as t_r
import geomadi.train_viz as t_v
import custom.lib_custom as l_c
import pickle

print('--------------------------------define------------------------')
custD = "mc"
custD = "tank"
idField = "id_poi"
version = "11u"
version = "prod"
modDir = baseDir + "raw/"+custD+"/model/"
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: str(x)+"T")
dateL.loc[:,"time"] = dateL["day"]
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi[poi['use'] == 3]

print('---------------------------------load-join------------------------------')
gact = pd.read_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",dtype={idField:str})
ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical = t_r.isocal2day(ical,dateL)
dirc = t_r.mergeDir(baseDir+"raw/"+custD+"/viaCount/")
#dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirCount/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
tist = t_e.joinSource(gact,ical,how="outer",idField=idField,isSameTime=False)
tist = t_e.concatSource(tist,dirc,how="left",idField=idField,varName="foot")
tist.loc[:,"foot"] = tist['foot']*0.4226623389831198
tist = tist[tist['time'] == tist['time']]
tist.loc[:,"day"] = tist['time'].apply(lambda x:x[:11])
tist = tist[tist["day"].isin(dateL["day"])]
tist = tist[tist['time'] > '2018-12-31T']
tL = ['day','use','wday','cloudCover','humidity','visibility','Tmax']
tist = pd.merge(tist,dateL[tL],on="day",how="left")
tist = tist.replace(float('nan'),0)
tist = tist[tist[idField].isin(poi[idField])]
tist = tist.sort_values([idField,'time'])
print(tist.head(2))
print("merged locations %d" % len(set(gact[idField])))
print("sum of days %d" % (len(set(tist['day']))) )
print('--------------------------------regressor-----------------------------')
tL, sL = l_c.prepareTank(poi)
tact = tact[tact[idField].isin(poi.loc[poi['use'] == 3,idField])]
tist.loc[:,"pred"] = 0.
for i,g in tist.groupby(idField):
    fName = baseDir + "train/"+custD+"/prod/poi_" + i + ".pkl"
    fit_w = pickle.load(open(fName,'rb'))
    X = g[sL].values
    y_pred = fit_w.predict(X)
    tist.loc[tist[idField] == i,"pred"] = y_pred

print('-------------------------sum-up-and-check---------------------------')
pact = tist.pivot_table(index=idField,columns="time",values="pred",aggfunc=np.sum).reset_index()
pact.to_csv(baseDir + "raw/"+custD+"/act_predict/act_predict_"+version+".csv.gz",compression="gzip",index=False)

if True:
    for i,g in tist.groupby(idField):
        tL = t_r.day2time(g['time'])
        plt.title("poi: %s" % i)
        plt.plot(tL,g['act'],label="weighted")
        plt.plot(tL,g['ref'],label="isocalendar")
        plt.plot(tL,g['pred'],label="prediction")
        plt.legend()
        plt.xticks(rotation=15)
        plt.show()
        break

