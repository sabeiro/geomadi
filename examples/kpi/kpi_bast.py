import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.train_reshape as t_r

print('--------------------------------define------------------------')
custD = "bast"
idField = "id_poi"
dateL = pd.read_csv(baseDir + "raw/"+custD+"/dateList.csv")
dateL.loc[:,"day"] = dateL['day'].apply(lambda x: x + "T")
via = pd.read_csv(baseDir+"raw/"+custD+"/dirc_via_feb_d.csv.gz",compression="gzip",dtype={idField:str})
dirc = pd.read_csv(baseDir+"raw/"+custD+"/dirCount_d.csv.gz",compression="gzip",dtype={idField:str})
ical = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_d.csv.gz",compression="gzip",dtype={idField:str})
ical.loc[:,idField] = ical[idField].apply(lambda x: str(int(float(x))))
ical = t_r.isocal2day(ical,dateL)
vist = t_e.joinSource(via,ical,how="left",idField=idField,isSameTime=False)
vist = t_e.concatSource(vist,dirc,how="left",idField=idField,varName="dirc")
vist.columns = [idField,"time","via","bast","dirc"]
corrV = 0.3505472737517056
corrF = 0.4115174710976628
vist.loc[:,"dirc"] = vist['dirc']*corrF
vist.loc[:,"via"]  = vist['via']*corrV
fist = vist.dropna()
fist = fist[fist['dirc']>0]

if False:
    print('---------------------remainder-distribution----------------------')
    t_v.plotJoin(vist,col_ref="bast",col1="via",col2="dirc")
    
    vist.loc[:,"via_r"] = vist['via'] - vist['bast']
    vist.loc[:,"dirc_r"] = vist['dirc'] - vist['bast']
    import seaborn as sns
    plt.title("remainder distribution")
    sns.kdeplot(vist['via_r'], shade=True, bw=.5, color="olive",label="via")
    sns.kdeplot(vist['dirc_r'], shade=True, bw=.05, color="purple",label="tile")
    plt.legend()
    plt.xlabel("deviation")
    plt.ylabel("density")
    plt.show()






print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
