import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import custom.lib_custom as t_l
import geomadi.series_stat as s_l
import geomadi.train_model as tlib
import geomadi.train_reshape as t_f
import geomadi.train_shape as t_s
import geomadi.train_score as t_c
import geomadi.train_convNet as t_k
import custom.lib_custom as l_c
import geomadi as gm

custD = "bast"
idField = "id_poi"
dirc = pd.read_csv(baseDir + "raw/"+custD+"/dirc_via_feb_h.csv.gz",compression="gzip",dtype={idField:str})
hL = t_r.timeCol(dirc)
iL = t_r.date2isocal(hL,date_format="%Y-%m-%dT%H:%M:%S")
dirc = dirc[[idField] + hL]
dirc.columns = [idField] + iL
refi = pd.read_csv(baseDir + "raw/"+custD+"/ref_iso_h.csv.gz",compression="gzip")

importlib.reload(t_r)
dlX = t_r.isocalInWeek(dirc,isBackfold=True)
dlY = t_r.isocalInWeek(refi,isBackfold=False)
XL = np.array([x['values'] for x in dlX])
YL = np.array([x['values'] for x in dlY])

print('--------------dictionary-learning-------------------')
from sklearn.cluster import KMeans
n_cluster = 2
clustM = {}
kpiD = {}
for n in [1,2,4,8,16,24]:
    print(n)
    clusterer = KMeans(copy_x=True,init='k-means++',max_iter=600,n_clusters=n,n_init=10,n_jobs=1,precompute_distances='auto',random_state=None,tol=0.0001,verbose=0)
    yL = np.reshape(YL,(len(YL),YL.shape[1]*YL.shape[2]))
    mod = clusterer.fit(yL)
    clustM[n] = clusterer
    centroids = clusterer.cluster_centers_
    nearest_idx = clusterer.predict(yL)
    kpiL = []
    for i in range(len(yL)):
        y1 = yL[i]
        y2 = centroids[nearest_idx[i]]
        e = t_c.relErr(y1,y2)
        kpiL.append({"id":i,"r":sp.stats.pearsonr(y1,y2)[0],"e":e[2],"d":(y1.mean()-y2.mean())/(y1.mean()+y2.mean())*2.})
    kpiL = pd.DataFrame(kpiL)
    kpiD[n] = kpiL

n = 24
centroids = clustM[n].cluster_centers_
np.save(baseDir+"raw/"+custD+"/dictionary.npy",np.reshape(centroids,(n,7,24)))
    
if False:
    poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv",dtype={idField:str})
    clustL = []
    for g in dlY:
        idx = clustM[2].predict([np.reshape(g['values'],(YL.shape[1]*YL.shape[2]))])[0]
        clustL.append({idField:g[idField],"week":g['week'],"commuter":idx})
    clustL = pd.DataFrame(clustL)
    clustP = clustL.groupby(idField).agg(np.mean).reset_index()
    clustW = clustL.groupby("week").agg(np.mean).reset_index()
    t_v.plotHistogram(clustP['commuter'],nbin=20,label="commuter")
    plt.show()
    plt.bar(clustW['week'],clustW['commuter'])
    plt.xlabel("week number")
    plt.ylabel("commuting pattern")
    plt.show()
    
    poi.loc[:,"commuter"] = poi.merge(clustP,how="left",on=idField,suffixes=["_x",""])['commuter'].values
    poi.to_csv(baseDir + "raw/"+custD+"/poi.csv",index=False)
    
if False:
    print('----------------plot-histogram--------------------')
    bins = 20
    fig, ax = plt.subplots(1,2)
    for k in kpiD:
        ax[0].hist(kpiD[k]['r'],bins=bins,normed=1,histtype='step',cumulative=1,label=k)
    ax[0].set_xlabel("correlation")
    ax[0].set_ylabel("density")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    for k in kpiD:
        ax[1].hist(kpiD[k]['e'],bins=bins,normed=1,histtype='step',cumulative=-1,label=k)
    ax[1].legend()
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel("relative error")
    ax[1].set_ylabel("density")
    plt.show()
    
    fig, ax = plt.subplots(1,2)
    t_v.kpiDis(kpiD[1],tLab="cross valid",col_cor="r",col_dif="e",col_sum="r",isRel=True,ax=ax[0])
    t_v.kpiDis(kpiD[1],tLab="cross valid",col_cor="r",col_dif="d",col_sum="r",isRel=False,ax=ax[1])
    plt.show()

if False:
    print('-------------------plot-dictionary---------------------')
    n_row = 4
    n = 24
    n_col = min(18,int(n/n_row))
    lL = random.sample(range(len(yL)),len(yL))
    cmap = plt.get_cmap("viridis")
    centroids = clustM[n].cluster_centers_
    for i in range(n_row):
        for j in range(n_col):
            axes = plt.subplot(n_row,n_col,n_col*i+j+1)
            y = centroids[n_col*i+j]
            axes.get_xaxis().set_visible(False)
            axes.get_yaxis().set_visible(False)
            #plt.imshow(np.reshape(y,(YL.shape[1],YL.shape[2])),cmap=cmap)
            plt.imshow(np.reshape(y,(7,24)),cmap=cmap)
    plt.show()






