import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.series_stat as s_s

idField = "id_poi"
custD = "tank"
cred = json.load(open(baseDir + '/credenza/geomadi.json'))
metr = json.load(open(baseDir + '/raw/basics/metrics.json'))['metrics']

poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
# if custD == "tank":
#     poi = poi[poi['use'] == 3]

mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
mist.loc[:,idField] = mist[idField].astype(str)
mist = mist[mist[idField].isin(poi[idField])]
hL = mist.columns[[bool(re.search('-??T',x)) for x in mist.columns]]
hL = [x for x in hL if x >= '2019-02-00T']
mact = pd.melt(mist,value_vars=hL,id_vars=idField)
mact = mact.dropna()
mact.columns = [idField,"time","ref"]
mact.sort_values(idField,inplace=True)

import importlib
importlib.reload(s_s)

if False:
    print('-----------------------autocompatibility-reference-data---------------------')
    def clampF(x):
        return pd.Series({"r":sp.stats.pearsonr(x['ref'],x['ref1'])[0]})

    iL = np.unique(poi[idField])
    gacL = []
    noiseL = [.05,.1,.15,.2,.3,.4,.5,.6]
    #noiseL = [.05,.1,.15]
    fig, ax = plt.subplots(1,len(noiseL))
    for i,j in enumerate(noiseL):
        mact.loc[:,"ref1"] = s_s.addNoise(mact['ref'],noise=j)
        gact = mact.groupby(idField).apply(clampF)
        k = gact.loc[gact['r'] == min(gact['r']),"r"].index[0]
        gact.columns = ["noise_"+str(j)]
        gacL.append(gact)
        if False:
            g = mact[mact[idField] == k]
            r = sp.stats.pearsonr(g['ref'],g['ref1'])[0]
            ax[i].set_title("poi %s noise level %.2f corr %.2f" % (k,j,r) )
            ax[i].plot(g['ref1'].values,label="reference + noise")
            ax[i].plot(g['ref'].values,label="reference")
            ax[i].legend()

    gact = pd.concat(gacL,axis=1)
    plt.show()
    plt.title("autocompatibility reference data")
    gact.boxplot()
    plt.xlabel("noise level")
    plt.ylabel("correlation")
    plt.xticks(rotation=15)
    plt.show()

    

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
