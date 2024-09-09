#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import geomadi.lib_graph as gra
import geomadi.series_lib as s_l
import seaborn as sns
from sklearn.decomposition import PCA
import geomadi.train_keras as t_k

custD = "tank"
idField = "id_poi"
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
poi = poi[poi['use'] == 3]
mist = pd.read_csv(baseDir + "raw/"+custD+"/ref_visit_d.csv.gz",compression="gzip")
mist = mist[mist[idField].isin(poi[idField])]
hL = [x for x in mist.columns if bool(re.search("T",x))]

import importlib
importlib.reload(t_k)

XL, idL, den, norm = t_k.dayInWeek(mist)
y = []
for x in XL:
    y = np.append(y,x)
    
segment_len = 7
slide_len = 7
segments = []
for start_pos in range(0, len(y), slide_len):
    end_pos = start_pos + segment_len
    segment = np.copy(y[start_pos:end_pos])
    if len(segment) != segment_len:
        continue
    segments.append(segment)
    print(len(segments))
    window_rads = np.linspace(0, np.pi, segment_len)
    window = np.sin(window_rads)**2
    windowed_segments = []
for segment in segments:
    windowed_segment = np.copy(segment) * window
    windowed_segments.append(windowed_segment)

from sklearn.cluster import KMeans
n_cluster = 24
clusterer = KMeans(copy_x=True,init='k-means++',max_iter=600,n_clusters=n_cluster,n_init=10,n_jobs=1,precompute_distances='auto',random_state=None,tol=0.0001,verbose=2)
#clusterer.fit(windowed_segments)
mod = clusterer.fit(segments)

if True:
    corr = pd.DataFrame(np.corrcoef(np.array(centroids)))
    ax = sns.heatmap(corr, vmax=1, square=True,annot=False,cmap='RdYlGn')
    plt.show()

if True:
    centroids = clusterer.cluster_centers_
    nearest_idx = clusterer.predict(segments)
    kpiL = []
    for i in range(len(segments)):
        y1 = segments[i]
        y2 = centroids[nearest_idx[i]]
        kpiL.append({"id":i
                     ,"r":sp.stats.pearsonr(y1,y2)[0]
                     ,"v":np.sqrt(np.mean((y1-y2)**2))
                                  })
    kpiL = pd.DataFrame(kpiL)
    print("%.2f %% over 0.6" % (kpiL[kpiL['r']>0.6].shape[0]/kpiL.shape[0]))

    n_col = min(18,int(n_cluster/3))
    lL = random.sample(range(len(segments)),len(segments))
    for i in range(3):
        for j in range(n_col):
            axes = plt.subplot(3,n_col,n_col*i+j+1)
            l = lL[min(len(disp)-1,n_col*i+j)]
            y1 = segments[l]
            y2 = centroids[nearest_idx[l]]
            plt.title("corr %.2f rmse %.2f" % (sp.stats.pearsonr(y1,y2)[0],np.sqrt(np.mean((y1-y2)**2))) )
            plt.plot(y1,label="Original segment")
            plt.plot(y2,label="Nearest centroid")
    # plt.legend()
    # plt.tight_layout()
    plt.show()
    

if False:
    disp = segments
    disp = windowed_segments
    disp = clusterer.cluster_centers_
    plt.figure()
    lL = random.sample(range(len(disp)),len(disp))
    lL = range(n_cluster)
    n_col = min(18,int(n_cluster/3))
    for i in range(3):
        for j in range(n_col):
            axes = plt.subplot(3,n_col,n_col*i+j+1)
            l = lL[min(len(disp)-1,n_col*i+j)]
            plt.title("segment " + str(l))
            plt.plot(disp[l],label="segment " + str(l))
    # plt.legend()
    # plt.tight_layout()
    plt.show()

