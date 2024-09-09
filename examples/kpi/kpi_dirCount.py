import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.proc_lib as plib
import geomadi.series_stat as s_s
import scipy.stats as stats
import pymongo

print('---------------------------------direction-counts--------------------------')
custD = "mc"
custD = "bast"
custD = "tank"
idField = "id_poi"

cred = json.load(open(baseDir + "credenza/geomadi.json","r"))
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

print('----------------------------dir-count-vs-via-counts---------------------')
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
dirc = pd.read_csv(baseDir + "raw/tank/dirCount/dirCount_d.csv.gz",dtype={idField:str})
hL = t_r.timeCol(dirc)
dirm = dirc.melt(id_vars=idField,value_vars=hL)
via = pd.read_csv(baseDir + "raw/tank/viaCount/dirc_via_feb_d.csv.gz",dtype={idField:str})
hL = t_r.timeCol(via)
viam = via.melt(id_vars=idField,value_vars=hL)
dirm = dirm.merge(viam,on=[idField,"variable"],how="inner",suffixes=["_dirc","via"])
dirm.columns = [idField,"time","dirc","via"]
dirm = dirm.sort_values([idField,"time"])
dirm = dirm[dirm[idField].isin(poi.loc[poi['use']==3,idField])]
scor = t_s.scorPerf(dirm,step="via",col1="dirc",col2="via")
dirm.to_csv(baseDir+"gis/tank/via_dirc.csv",index=False)
scor = scor.merge(poi[[idField,'x','y']],on=idField,how="left")
scor.to_csv(baseDir+"gis/tank/via_dirc_scor.csv",index=False)

foot = pd.read_csv(baseDir+"raw/tank/foot.csv",dtype={idField:str})
dirp = dirm.groupby(idField).agg(np.mean).reset_index()
foot.loc[:,"dirc"] = foot.merge(dirp,on=idField,how="left")['dirc_y']
foot.loc[:,"via"] = foot.merge(dirp,on=idField,how="left")['via_y']
foot = foot.sort_values(idField)
foot.to_csv(baseDir+"raw/tank/foot.csv",index=False)

if False:
    print('-------------------plot-deviations-from-bast----------------')
    y_inter = s_s.linReg(foot['via']*corrO,foot['bast']*.5)
    corrO = 0.4226623389831198#0.464928
    corrF = foot['bast'].mean()*.5/foot['via'].mean()
    plt.scatter(foot['via']*corrO,foot['dirc']*corrO,label="via vs tile",alpha=.5,marker="s")
    plt.plot(foot['dirc']*corrO,foot['dirc']*corrO,label="intercept")
    plt.scatter(foot['via']*corrO,foot['bast']*.5,label="via vs bast")
    plt.scatter(foot['via']*corrF,foot['bast']*.5,label="via corr vs bast")
    plt.plot(foot['via']*corrO,y_inter,label="intercept corr")
    plt.scatter(foot['dirc']*corrO,foot['bast']*.5,label="tile vs bast",alpha=.5,marker="s")
    plt.legend()
    plt.show()
    
    print('-------------------deviation-boxplot---------------------')
    foot.loc[:,"dirc_bast"] = foot['dirc']*corrO-foot['bast']*.5
    foot.loc[:,"via_bast"]  = foot['via']*corrO -foot['bast']*.5
    foot.loc[:,"via_bast_n"]= foot['via']*corrF -foot['bast']*.5
    foot.boxplot(column=['dirc_bast','via_bast','via_bast_n'])
    plt.show()

if False:
    print('------------------bar-plot----------------------')
    foot = foot.sort_values("dirc_bast")
    l = np.array(range(foot.shape[0]))
    fig, ax = plt.subplots(1,1)
    ax.bar(l,foot['dirc_bast'],width=.4,label="tile",alpha=.5)
    ax.bar(l+.50,foot['via_bast_n'],width=.4,label="via",alpha=.5)
    ax.legend()
    ax.set_xticklabels(cist[idField].values)
    plt.ylabel('deviation')
    plt.xlabel('id_poi')
    plt.show()

    
if False:
    print('-----------------------single-location-curves-----------------')
    i = '1251'
    g = dirm[dirm[idField] == i]
    for i,g in dirm.groupby(idField):
        plt.title("location %s" %i)
        plt.plot(g['dirc'],label="tile")
        plt.plot(g['via'],label="via")
        plt.legend()
        plt.show()

if False:
    print('-------------------------show-local-densities-------------------')
    import joypy
    fig, axes = joypy.joyplot(dirm,column=['dirc','via'],by=idField,ylim='own',figsize=(12,6),alpha=.5)#,colormap=cm.summer_r)
    plt.legend()
    plt.title('density distribution of activity (blue:tile, orange: via)')
    plt.show()
    
    dirm.boxplot(column=['dirc','via'])
    plt.show()

    fig, ax = plt.subplots(1,1)
    bx1 = dirm.boxplot(column=["dirc"],by=idField,ax=ax,return_type="dict")
    [[item.set_color('g') for item in bx1[key]['boxes']] for key in bx1.keys()]
    bx3 = dirm.boxplot(column=['via'],by=idField,ax=ax,return_type="dict")
    [[item.set_color('o') for item in bx2[key]['boxes']] for key in bx2.keys()]
    plt.xticks(rotation=15)
    plt.show()

    dirm.boxplot(column=['dirc','via'],by=idField)
    plt.show()

