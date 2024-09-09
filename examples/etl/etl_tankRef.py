import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
import pandas as pd
import numpy as np
import datetime
import shutil
from dateutil import tz
import geomadi.train_shapeLib as t_s

from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Europe/Berlin')

if True: ##yearly reference values
    tar = tarfile.open(baseDir+"log/"+custD+"/ref_visit.tar.gz","r:gz")
    tar.extractall(path="/tmp/")
    tar.close()
    fileL = os.listdir("/tmp/ref_year")
    gist = pd.DataFrame()
    for f in fileL:
        vist = pd.read_csv("/tmp/ref_year/"+f).fillna(method='ffill')
        totR = vist.apply(lambda x: 'Ergebnis' in x.values,axis=1)
        vist = vist[~totR]
        vist = vist.replace("#",24)
        vist.loc[:,"Verkaufsstunde"] = vist["Verkaufsstunde"].apply(lambda x: "%02d" % (int(x)-1))
        vist.loc[:,"time"] = vist[['Kalendertag','Verkaufsstunde']].apply(lambda x: x[0][6:] + "-" + x[0][3:5] + "-" + x[0][:2] + "T" + x[1] + ":00:00",axis=1)
        vist.loc[:,"Wirtschaftseinheit"] = vist["Wirtschaftseinheit"].astype(int)
        vist.loc[:,"Anzahl Bons"] = vist["Anzahl Bons"].apply(lambda x: re.sub(",","",str(x)))        
        vist.loc[:,"Anzahl Bons"] = vist["Anzahl Bons"].astype(float)
        vist = vist[["Wirtschaftseinheit","time","Anzahl Bons"]]
        vist.columns = ["id_poi","time","ref"]
        gist = pd.concat([gist,vist],axis=0)
    gist = gist.groupby(['id_poi','time']).agg(sum).reset_index()
    pist = gist.pivot_table(index="id_poi",values="ref",columns="time",aggfunc=np.sum).reset_index()
    pist.to_csv(baseDir + "raw/tank/ref_visit_h.csv.gz",compression="gzip",index=False)
    gist.loc[:,"day"] = gist['time'].apply(lambda x: x[:10]+"T")
    dist = gist[['id_poi','day','ref']].groupby(['id_poi','day']).agg(sum).reset_index()
    pist = dist.pivot_table(index="id_poi",values="ref",columns="day",aggfunc=np.sum).reset_index()
    pist.to_csv(baseDir + "raw/tank/ref_visit_d.csv.gz",compression="gzip",index=False)
    shutil.rmtree("/tmp/ref_year")

if False:
    plog('----------------yearly reference values-cash-id-------------------------')
    fileL = os.listdir(baseDir + "log/tank/ref_year")
    fileL = [x for x in fileL if bool(re.search("_17",x))]
    f = fileL[0]
    gist = pd.DataFrame()
    for f in fileL:
        vist = pd.read_csv(baseDir + "log/tank/ref_year/" + f).fillna(method='ffill')
        totR = vist.apply(lambda x: 'Ergebnis' in x.values,axis=1)
        vist = vist[~totR]
        vist = vist.replace("#",24)
        vist.loc[:,"Verkaufsstunde"] = vist["Verkaufsstunde"].apply(lambda x: "%02d" % (int(x)-1))
        vist.loc[:,"time"] = vist[['Kalendertag','Verkaufsstunde']].apply(lambda x: x[0][6:] + "-" + x[0][3:5] + "-" + x[0][:2] + "T" + x[1] + ":00:00",axis=1)
        vist.loc[:,"Wirtschaftseinheit"] = vist["Wirtschaftseinheit"].astype(int)
        vist.loc[:,"Anzahl Bons"] = vist["Anzahl Bons"].astype(float)
        vist = vist[["Wirtschaftseinheit","Betrieb","time","Anzahl Bons"]]
        vist.columns = ["id_poi","id_cash","time","ref"]
        vist = vist[vist['id_poi'].isin(corrL['id_poi'])]
        gist = pd.concat([gist,vist],axis=0)

    gist.to_csv(baseDir + "raw/tank/ref_cash.csv",index=False)
    pist = gist.pivot_table(index=["id_poi","id_cash"],values="ref",aggfunc=np.mean).reset_index()
    pist = pist[pist['Wirtschaftseinheit'].isin(corrL['id_poi'])]

