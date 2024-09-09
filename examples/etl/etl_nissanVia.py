#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import pymongo
import geomadi.kernel_lib as k_l
from scipy import signal as sg
import geomadi.proc_lib as plib

def plog(text):
    print(text)

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))

def deLoop1(locV):
    en, ex = "998", "998"
    enV = [x for x in locV if bool(re.search("entry",x))]
    exV = [x for x in locV if bool(re.search("exit",x))]
    if len(enV) > 0:
        en = enV[0]
    if len(exV) > 0:
        ex = exV[-1]
    if en == ex:
        en, ex = "998", "998"
    if any([ex == x for x in enV]):
        ex = "998"
    if any([en == x for x in exV]):
        en = "998"
    return en, ex

def deLoop2(locV):
    en, ex = '998', '998'
    if len(locV) < 1:
        return en, ex
    if len(locV) == 1:
        if bool(re.search("entry",locV[0])):
            en = locV[0].split("_")[1]
        else:
            ex = locV[-1].split("_")[1]
    else :
        if bool(re.search("entry",locV[-1])):
            en = locV[-1].split("_")[1]
        if bool(re.search("exit",locV[0])):
            en = locV[1].split("_")[1]
    return en, ex

def deLoop3(locV):
    en, ex = '998', '998'
    if len(locV) < 1:
        return en, ex
    mask = [True for x in range(len(locV))]
    for j in range(len(locV)-1):
        if locV[j].split("_")[1] == locV[j+1].split("_")[1]:
            mask[j] = False
            mask[j+1] = False
            #nLoop += 1
    junL = [x for x,y in zip(locV,mask) if y]
    if len(junL) < 1:
        return en, ex
    entryL = [x for x in junL if bool(re.search("entry",x))]
    exitL  = [x for x in junL if bool(re.search("exit",x))]
    if len(entryL) > 0:
        en = entryL[0]
    if len(exitL) > 0:
        ex = exitL[-1]
    return en, ex

nodeD = pd.read_csv(baseDir + "raw/nissan/en2ex_dist.csv")
nodeD.loc[:,"j_en"] = nodeD['en'].apply(lambda x: re.sub("entry_","",x))
nodeD.loc[:,"j_en"] = nodeD['j_en'].apply(lambda x: re.sub("[abc]","",x))
nodeD.loc[:,"j_ex"] = nodeD['ex'].apply(lambda x: re.sub("exit_","",x))
nodeD.loc[:,"j_ex"] = nodeD['j_ex'].apply(lambda x: re.sub("[abc]","",x))
nodeD = nodeD[['dist','j_en','j_ex']].groupby(["j_en","j_ex"]).agg(np.mean).reset_index()
nodeD.loc[:,"dist"] = nodeD['dist']/1000.
#nodeD.loc[:,"j_en"] = nodeD['j_en'].astype(int)
#nodeD.loc[:,"j_ex"] = nodeD['j_ex'].astype(int)
nodeD.to_csv(baseDir + "raw/nissan/en2ex_dist_int.csv",index=False)

fileL = ["_v1","_v2","_v3","_v4"]
fileL = ["_v6"]

for vi in fileL:
    via = pd.read_csv(baseDir + "raw/nissan/via_via"+vi+".csv")
    via.loc[:,"locV"] = via['location'].apply(lambda x: x.split(";"))
    N = via['count'].sum()
    # setL = [bool(re.search("^entry",x)) for x in via['location']]
    # via = via[setL]
    via.loc[:,"enter"] = 'entry_998'
    via.loc[:,"exit"] = 'exit_998'
    nLoop = 0
    for i,g in via.iterrows():
        via.loc[i,"enter"], via.loc[i,"exit"] = deLoop1(via.loc[i,'locV'])
        
    print("no enter: lost %.2f%%" % (via[  [x == '998' for x in via['enter']]   ]['count'].sum()/N))
    print("no exit : lost %.2f%%" % (via[  [x == '998' for x in via['exit']]   ]['count'].sum()/N))
    setL = [len(x) > 2 for x in via['locV']]
    print("#loops: %.2f%%" % (via[setL]['count'].sum()/via['count'].sum()) )

    viaP = via.pivot_table(index=["enter"],columns="exit",values="count",aggfunc=np.sum)
    viaP.replace(float('nan'),0,inplace=True)
    if False:
        viaP = viaP.iloc[:,[bool(re.search("exit",x)) for x in viaP.columns]]
        viaP = viaP[[bool(re.search("entry",x)) for x in viaP.index]]
    viaP.index = [re.sub("entry_","",x) for x in viaP.index]
    #viaP.index = [int(re.sub("[abc]","",x)) for x in viaP.index]
    viaP.columns = [re.sub("exit_","",x) for x in viaP.columns]
    #viaP.columns = [int(re.sub("[abc]","",x)) for x in viaP.columns]
    # viaP = viaP[~viaP.index.isin(['45','56'])]
    # viaP = viaP.iloc[:,~viaP.columns.isin(['45','56'])]
    viaT = pd.melt(viaP.reset_index(),value_vars=viaP.columns,id_vars="index")
    viaT.columns = ["enter","exit","count"]

    if False: #996 to junctions over 150km
        nodeD = nodeD[['dist','j_en','j_ex']].groupby(["j_en","j_ex"]).agg(np.mean).reset_index()
        nodeD1 = nodeD[['j_ex','j_en','dist']]
        nodeD1.columns = ['j_en','j_ex','dist']
        nodeD = pd.concat([nodeD,nodeD1],axis=0)
        nodeD = nodeD.groupby(["j_en","j_ex"]).agg(np.mean).reset_index()
        viaT = pd.merge(viaT,nodeD,left_on=["enter","exit"],right_on=["j_en","j_ex"],how="left")
        viaT.loc[viaT['dist'] > 150,"exit"] = 996
        
    viaP = viaT.pivot_table(index="enter",columns="exit",values="count",aggfunc=np.sum)
    viaP.replace(float('nan'),0,inplace=True)
    viaP.to_csv(baseDir + "raw/nissan/enter2exit"+vi+".csv")

if False:
    cmap = plt.get_cmap("PiYG") #BrBG
    T = viaP[viaP.columns[1:8]][viaP.index.isin(viaP.index[1:8])]
    sns.heatmap(T,cmap=cmap,linewidths=.0)
    plt.show()

if False:    
    job = json.load(open(baseDir + "src/job/odm_via/qsm.odm_extraction.odm_via_via.json"))
    locL = pd.DataFrame(job['odm_via_conf']['input_locations'])
    locL.loc[:,"id"] = locL['node_list'].apply(lambda x: x[0])
    nodeL = pd.read_csv(baseDir + "raw/nissan/junct_loc.csv")
    nodeL.loc[1,['x','y']] = nodeL.loc[0,['x','y']]
    viaEx = pd.melt(viaP)
    viaEx = viaEx.groupby('exit').agg(sum).reset_index()
    viaEn = pd.melt(viaP.T)
    viaEn = viaEn.groupby('enter').agg(sum).reset_index()
    viaEx = pd.merge(viaEx,nodeL,left_on="exit",right_on="loc",how="left")
    viaEn = pd.merge(viaEn,nodeL,left_on="enter",right_on="loc",how="left")
    viaEx.to_csv(baseDir + "raw/nissan/enter_count.csv",index=False)
    viaEn.to_csv(baseDir + "raw/nissan/exit_count.csv",index=False)

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

