#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

def plog(text):
    print(text)

import pymongo
with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)
    
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])

if False:
    odmS = pd.read_csv(baseDir + "log/nissan/odm_shortbreak.csv.tar.gz",compression="gzip")
    odmL = pd.read_csv(baseDir + "log/nissan/odm_longbreak.csv.tar.gz",compression="gzip")
    odmM = pd.merge(odmS,odmL,left_on=["origin","destination"],right_on=["origin","destination"],how="outer")
    odmM.loc[:,"diff"] = odmM['count_x'] - odmM['count_y']
    odmM.to_csv(baseDir + "log/nissan/odm_midbreaks.csv.tar.gz",compression="gzip",index=False)
    odmM.to_csv(baseDir + "log/nissan/odm_midbreaks.csv",index=False)

odmM  = pd.read_csv(baseDir + "log/nissan/odm_midbreaks.csv.tar.gz",compression="gzip")
zip2  = pd.read_csv(baseDir + "gis/graph/zip2zip_sel.csv")
nodeD = pd.read_csv(baseDir + "raw/nissan/nodeList.csv")
nodeD.loc[:,"grp"] = nodeD['grp'].apply(lambda x: re.sub("[a-z]","",str(x)))
nodeD = nodeD.groupby('id_junct').first().reset_index()
nodeD = nodeD[nodeD['highway'] == "motorway_link"]
zipG = zip2.groupby(["start_zip","end_zip","first_junction","last_junction"]).agg(np.mean).reset_index()
zipG.loc[:,"enter"] = pd.merge(zipG,nodeD,left_on="first_junction",right_on="id_junct",how="left")["grp"]
zipG.loc[:,"exit"] = pd.merge(zipG,nodeD,left_on="last_junction",right_on="id_junct",how="left")["grp"]
zipG = pd.merge(zipG,odmM,left_on=["start_zip","end_zip"],right_on=["origin","destination"],how="left")
del zipG['destination'],zipG['origin'], zipG['first_junction'], zipG['last_junction']
zipG = zipG[zipG['count_x'] + zipG['count_y'] > 0]
zipG.loc[zipG['enter'] != zipG['enter'],"enter"] = "998"
zipG.loc[zipG['exit'] != zipG['exit'],"exit"] = "998"
zipG.loc[zipG['enter'] == "","enter"] = "998"
zipG.loc[zipG['exit'] == "","exit"] = "998"
zipG.loc[zipG['motor_len']>150.,"exit"] = "996"
zipE = zipG[(zipG['enter'] == "998") | (zipG['exit'] == "998")]
def clampF(x):
    return pd.Series({"start_zip":"998","end_zip":"998","length":np.mean(x['length']),"motor_len":np.mean(x['motor_len']),"prim_len":np.mean(x['prim_len']),"sec_leng":np.mean(x['sec_leng']),"count_x":sum(x['count_x']),"count_y":sum(x['count_y']),"diff":sum(x['diff'])})
zipE = zipE.groupby(['enter','exit']).apply(clampF).reset_index()
zipE = zipE[zipG.columns]
zipG = zipG[(zipG['enter'] != "998")]
zipG = zipG[(zipG['exit']  != "998")]
zipG = pd.concat([zipG,zipE],axis=0)
zipG1 = zipG
sufS = ""
if True: #force symmetry
    sufS = "_sym"
    zipG = zipG1
    odmM.loc[:,"id_pair"] = odmM[['origin','destination']].apply(lambda x: str(x[0]) + "-" + str(x[1]),axis=1)
    zipG.loc[:,"pair_l"] = leftL  = zipG[['start_zip','end_zip']].apply(lambda x: str(x[0]) + "-" + str(x[1]),axis=1)
    zipG.loc[:,"pair_r"] = rightL = zipG[['end_zip','start_zip']].apply(lambda x: str(x[0]) + "-" + str(x[1]),axis=1)
    leftL  = np.unique(leftL[~leftL.isin(rightL)])
    rightL = np.unique(rightL[~rightL.isin(leftL)])
    id_pair = odmM[['origin','destination']].apply(lambda x: str(x[0]) + "-" + str(x[1]),axis=1)
    id_l = odmM.loc[odmM['id_pair'].isin(leftL),"id_pair"]
    id_r = odmM.loc[odmM['id_pair'].isin(rightL),"id_pair"]
    count_l = odmM.loc[odmM['id_pair'].isin(leftL),"count_y"]
    count_r = odmM.loc[odmM['id_pair'].isin(rightL),"count_y"]
    zipL = zipG[zipG['pair_l'].isin(id_r)]
    swapL = zipL['enter']
    zipL.loc[:,"enter"] = zipL['exit']
    zipL.loc[:,"exit"] = swapL.values
    zipL.loc[:,"count_y"] = count_l
    zipR = zipG[zipG['pair_r'].isin(id_l)]
    if zipR.shape[0] > 0:
        swapL = zipR['enter']
        zipR.loc[:,"enter"] = zipR['exit']
        zipR.loc[:,"exit"] = swapL
        zipR.loc[:,"count_y"] = count_r
    zipG = pd.concat([zipG,zipR,zipL],axis=0)
    zipG = zipG[zipG['enter'] != zipG['exit']]


zipG.loc[:,"count"] = zipG['count_y']*zipG['motor_len']/zipG['motor_len'].max()
zipP = zipG.pivot_table(index="enter",columns="exit",values="count",aggfunc=np.nansum).replace(np.nan,0)
zipP.to_csv(baseDir + "raw/nissan/enter2exit_short"+sufS+".csv")
zipG.pivot_table(index="enter",columns="exit",values="diff",aggfunc=np.nansum).replace(np.nan,0).to_csv(baseDir + "raw/nissan/enter2exit_break"+sufS+".csv")
zipG.to_csv(baseDir + "raw/nissan/zip_group"+sufS+".csv",index=False)

if False:
    zipM = zipG.pivot_table(index="enter",columns="exit",values="length",aggfunc=np.nansum).replace(np.nan,0).reset_index()
    zipM = pd.melt(zipM,id_vars="enter",value_vars=zipM.columns[1:])
    zipM.loc[:,"pair"] = zipM.apply(lambda x: "%s-%s" % (min(x['enter'],x['exit']),max(x['enter'],x['exit'])),axis=1)
    zipM = zipM[['pair','value']].groupby('pair').agg(np.mean).reset_index()
    zipM.columns = ['pair','distance']
    zipM1 = zipM.copy()
    zipM1.loc[:,"pair"] = zipM1['pair'].apply(lambda x: "%s-%s" % (x.split("-")[1],x.split("-")[0]))
    zipM = pd.concat([zipM,zipM1],axis=0)
    zipM = zipM[['pair','distance']].groupby('pair').agg(np.mean).reset_index()
    zipM.loc[:,"pair"] = zipM['pair'].apply(lambda x: "entry_%s-exit_%s" % (x.split("-")[1],x.split("-")[0]))

    grpD = nodeD[['grp','x','y']].groupby("grp").agg(np.mean).reset_index()
    from scipy.spatial.distance import squareform, pdist
    zipT = pd.DataFrame(squareform(pdist(grpD.iloc[:,1:])),columns=grpD['grp'],index=grpD['grp']).reset_index()
    zipT.columns.name = "exit"
    zipT = pd.melt(zipT,id_vars="grp",value_vars=zipT.columns[1:])
    zipT.loc[:,"pair"] = zipT.apply(lambda x: "%s-%s" % (x['grp'],x['exit']),axis=1)
    zipT = zipT[['pair','value']].groupby('pair').agg(np.mean).reset_index()
    zipT.loc[:,"pair"] = zipT['pair'].apply(lambda x: "entry_%s-exit_%s" % (x.split("-")[1],x.split("-")[0]))
    zipT.columns = ['pair','angle_dif']
    zipM = pd.merge(zipM,zipT,on="pair",how="outer")
    zipM.to_csv(baseDir + "raw/nissan/enter2exit_dist.csv",index=False)

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
