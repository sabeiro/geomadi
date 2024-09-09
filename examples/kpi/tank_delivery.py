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
import joypy
from matplotlib import cm

print('--------------------------------define------------------------')
custD = "tank"
idField = "id_poi"
version = "11u"
version = "prod"
modDir = baseDir + "raw/"+custD+"/model/"
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: str(x)+"T")
dateL.loc[:,"time"] = dateL["day"]
dateL = dateL[dateL['day'] > '2018-12-31T']
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = poi[poi['competitor'] == 0]
poi = poi[poi['use'] == 3]
corrF = 0.3181321540397981
corrO = 0.4226623389831198#0.464928
corP = poi[[idField]]
corP.loc[:,"cor_foot"] = 1.
corP.loc[corP[idField] == '1351',"cor_foot"] = [5.559189]

print('---------------------------------load-join------------------------------')
pact = pd.read_csv(baseDir + "raw/"+custD+"/act_predict/act_predict_"+version+".csv.gz",compression="gzip",dtype={idField:str})
wact = pd.read_csv(baseDir + "raw/"+custD+"/act_weighted/act_weighted_"+version+".csv.gz",compression="gzip",dtype={idField:str})
ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical = t_r.isocal2day(ical,dateL)
dirc = t_r.mergeDir(baseDir+"raw/"+custD+"/viaCount/")
vist = t_e.joinSource(pact,ical,how="outer",idField=idField,isSameTime=False)
vist = t_e.concatSource(vist,wact,how="left",idField=idField,varName="weighted")
vist = t_e.concatSource(vist,dirc,how="left",idField=idField,varName="foot")
vist.loc[:,"foot"] = vist['foot']*corrF
vist = vist[vist['time'] == vist['time']]
vist = vist[vist[idField].isin(poi[idField])]
vist = vist.sort_values([idField,'time'])
vist.loc[:,"day"] = vist['time'].apply(lambda x:x[:11])
vist = vist.merge(corP,on=idField,how="left")
vist.loc[:,"foot"] = vist['foot']*vist['cor_foot']
vist.loc[:,"deli"] = vist['act']*.5 + vist['weighted']*.5
vist.loc[:,"capt"] = vist['deli']/vist['foot']*100.
vist.loc[:,"month"] = vist['day'].apply(lambda x: x[5:7])
fist = vist[[idField,"day","deli","foot"]]
fist.columns = [idField,"day","act","direction_rate"]
fist.to_csv(baseDir + "out/act_foot_19_04.csv")

