#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.proc_lib as plib
from io import StringIO
def plog(text):
    print(text)

crmF = json.load(open(baseDir + "raw/basics/crmFact.json"))
ageC = pd.DataFrame(crmF['age'])
genC = pd.DataFrame(crmF['gender'])
dateL = os.listdir(baseDir + "log/destatis/")
dateL = [x for x in dateL if bool(re.search("tar.gz",x))]
for j in dateL:
    ageG = pd.read_csv(baseDir + "log/destatis/" + j,compression="gzip")
    ageG.columns = ["domzone"] + list(ageG.columns[1:])
    ageG = ageG[ageG['domzone']>=0]
    ageG = ageG[ageG['time'].isin(['22:00:00+00:00','23:00:00+00:00','00:00:00+00:00+1','01:00:00+00:00+1'])]
    ageG.loc[:,"time"] = ageG['time'].apply(lambda x: (int(x[:2]) - 2) % 24)
    for i in ageG.columns:
        ageG.loc[:,i] = ageG[i].astype(int)
    if bool(re.search("age",j)):
        ageG = ageG[ageG['age']>=0]
        ageG = ageG[ageG['gender']>=0]
        ageG = pd.merge(ageG,ageC,on="age",how="left")
        #ageG = pd.merge(ageG,genL,on="gender",how="left")
        ageG.loc[:,"count"] = ageG['count']*ageG['corr']
        gage = ageG[['domzone','age','gender','count']].groupby(['domzone','age','gender']).agg(np.mean).reset_index()
    elif bool(re.search("mcc",j)):
        ageG = ageG[ageG['mcc']>=0]
        mcc = pd.read_csv(baseDir + "raw/basics/mcc.csv")
        ageG = pd.merge(ageG,mcc,on="mcc",how="left")
        ageG = ageG[~(ageG['tourist'] == "Germany")]
        gage = ageG[['domzone','tourist','count']].groupby(['domzone','tourist']).agg(np.mean).reset_index()
    else :
        gage = ageG.groupby(['domzone']).agg(np.mean).reset_index()
    gage.to_csv(baseDir + "raw/destatis/" +  re.sub("\.tar\.gz","",j),index=False)

if False:
    import importlib
    import shutil
    importlib.reload(plib)
    dateL = os.listdir(baseDir + "log/destatis/commuter/")
    nrw = pd.read_csv(baseDir + "gis/geo/agsMix_bundesland.csv")
    nrw = nrw[['AGS','GEN']]
    for i in dateL:
        nameF = i.split(".")[0]
        nameF = "_".join(nameF.split("_")[2:])
        if True:
            ageG, fL = plib.parseTarPandas(baseDir + "log/destatis/commuter/",i,patterN="part-00000")
        else:
            ageG = pd.read_csv(baseDir+"log/destatis/"+'NRW_Pendler.csv.tar.gz',compression="gzip")
        ageG.columns = ["domzone"] + list(ageG.columns[1:])
        ageG = ageG[ageG['domzone']>=0]
        ageG = ageG[ageG['home_zone']>=0]
        ageG = ageG[ageG['last_zone']>=0]
        ageG = ageG[ageG['count'] > 0]
        ageG.loc[:,"domzone"] = ageG['domzone'].astype(int)
        ageG.loc[:,"home_zone"] = ageG['home_zone'].astype(int)
        ageG.loc[:,"last_zone"] = ageG['last_zone'].astype(int)
        ageG.loc[:,"count"] = ageG['count'].astype(int)
        ageG = ageG[ageG['domzone'] != ageG['home_zone']]
        ageG = ageG[ageG['last_zone'] == ageG['home_zone']]
        ageG.drop(columns=['time'],inplace=True)
        ageG = pd.merge(ageG,nrw,left_on="domzone",right_on="AGS",how="left")
        ageG = ageG[ageG['GEN'] == 'Nordrhein-Westfalen']
        #ageG = ageG[ageG['count'] > 30]
        ageG = ageG[ageG['aggregated_duration']>0]
        if any([x for x in ageG.columns if bool(re.search('age',x))]):
            ageG = pd.merge(ageG,ageC,on="age",how="left")
            ageG.loc[:,"count"] = ageG['count']*ageG['corr']
            del ageG['corr'], ageG['age']            
        if any([x for x in ageG.columns if bool(re.search('gender',x))]):
            ageG = pd.merge(ageG,genC,on="gender",how="left")
            ageG.loc[:,"count"] = ageG['count']*ageG['corr']
            del ageG['corr'], ageG['gender']
        ageR = ageG[['domzone','home_zone','count']].groupby(['domzone','home_zone']).agg(sum).reset_index()
        ageG.loc[:,"share"] = pd.merge(ageG,ageR,on=["domzone","home_zone"],how="left")["count_y"].values
        ageG.loc[:,"share"] = ageG['count']/ageG['share']
        del ageG['AGS'], ageG['GEN'], ageG['aggregated_duration'], ageG['last_zone']
        #ageG = ageG[['domzone','home_zone','count']]
        ageG.to_csv(baseDir+"raw/destatis/"+nameF+".csv",index=False)
        shutil.rmtree(fL[0].split("/")[0])

df1 = pd.read_csv(baseDir + "raw/destatis/destatis_NRW_Pendler.csv")
df2 = pd.read_csv(baseDir + "raw/destatis/destatis_NRW_Pendler_age.csv")
df = pd.merge(df1,df2,on=["domzone","home_zone"],how="outer")
print(df[df['count_x'] != df['count_x']])
    

        
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
