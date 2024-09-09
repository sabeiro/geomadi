import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.interpolate as interpolate
from multiprocessing.dummy import Pool as ThreadPool 
import scipy.ndimage as ndi
from scipy.interpolate import griddata
#from mpl_toolkits.basemap import Basemap
from scipy.spatial.distance import pdist, wminkowski, squareform
import geopandas as gpd
import shapely as sh
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

fs = json.load(open(os.environ['LAV_DIR'] + 'gis/geo/germany_border.geojson'))['features']
ags5 = gpd.read_file(baseDir + "gis/geo/bundesland.shp")
spotL = pd.read_csv(baseDir + 'raw/mc/mc_whitespot2.csv')
tileP = pd.read_csv(baseDir + 'raw/basics/tilePos.csv.gz')
tileI = pd.read_csv(baseDir + 'raw/mc/tile_isochrone.csv.gz')
tileI.loc[:,"isochrone"] = True
# pred = pd.read_csv(baseDir + 'raw/mc/daily_prediction.csv')
# pred.to_csv(baseDir + 'raw/mc/daily_prediction.csv.gz',compression="gzip",index=False)
pred = pd.read_csv(baseDir + 'raw/mc/daily_prediction.csv.gz')
pred = pred[pred['payslips_daily_avg_pred'] > 1200.]
pred = pred.merge(tileI,on="id_tile",how="left")
pred.loc[ pred['isochrone'] != pred['isochrone'],'isochrone'] = False
pred = pred[pred['isochrone']]
del pred['isochrone']
pred = pred.merge(tileP,on="id_tile",how="left")
spotL = pred

region = ags5['geometry']
for i,g in spotL.iterrows():
    point = sh.geometry.Point(g['x'],g['y'])
    setL = region.contains(point)
    if sum(setL) <= 0: continue
    idx = region[setL].index[0]
    spotL.loc[i,"ags"] = ags5.loc[idx,"state"]

i = '05770'
i = 'Niedersachsen'
g = spotL[spotL['ags'] == i]
spotL.loc[:,'cluster'] = float('nan')
centL = []
for i,g in spotL.groupby("ags"):
    #mDist1 = pdist(spotL[['XCenter','YCenter']],wminkowski,2,spotL['GP_max'])
    #mDist1 = pdist(spotL[['XCenter','YCenter']],metric="euclidean")
    setL = spotL['ags'] == i
    if g.shape[0] <= 10:
        clu = [0 for x in range(g.shape[0])]
    else:
        Z = linkage(g[['x','y']],'ward')
        clustD = 0.75
        clu = fcluster(Z, clustD, criterion='distance')#k, criterion='maxclust')
    print("%s tile/cluster (%d) ratio %.2f" % (i,len(set(clu)),g.shape[0]/len(set(clu))))
    spotL.loc[setL,'cluster'] = ["%s-%d" % (i,x) for x in clu]
    for j,gg in spotL[setL].groupby('cluster'):
        xM, yM = gg['x'].mean(), gg['y'].mean()
        p = gg['payslips_daily_avg_pred'].mean()
        a = gg['activities_daily_avg'].mean()
        d = gg['dircnt_daily_avg'].mean()
        gg.loc[:,"dist"] = np.sqrt( (gg['x'] - xM)**2 + (gg['y'] - yM)**2)
        idx = gg[gg['dist'] == min(gg['dist'])].index[0]
        c = gg.loc[idx]
        centL.append({"id_tile":c['id_tile'],'x':c['x'],'y':c['y'],'x_cluster':xM,'y_cluster':yM,'pred':p,'act':a,'dir':d,'n_tile':gg.shape[0],'cluster':j})

centL = pd.DataFrame(centL)

spotL.to_csv(baseDir + "gis/mc/daily_pred.csv",index=False)
centL.to_csv(baseDir + "gis/mc/whitespot_clust.csv",index=False)

spotL.to_csv(baseDir + "raw/mc/daily_pred.csv.gz",index=False,compression="gzip")
centL.to_csv(baseDir + "raw/mc/whitespot_clust.csv.gz",index=False,compression="gzip")


polyN = {}
for i,m in enumerate(fs):
    poly = np.array(m['geometry']['coordinates'])
    pn = np.array(poly[0][0])
    pn = np.append(pn,np.zeros((len(pn),1)),axis=1)
    polyN[i] = pn

fig, ax = plt.subplots()
for i in range(len(polyN)):
    ax.plot(polyN[i][:,0],polyN[i][:,1],c='g',alpha=0.5)
    
ax.scatter(spotC['x'],spotC['y'],s=200,c='r',alpha=0.3)
ax.scatter(spotL['x'],spotL['y'],c='b',alpha=0.5)
ax.scatter(spotM['x'],spotM['y'],s=40,c='y',alpha=0.5)
plt.show()

mDist = squareform(mDist1)

if False:
    plt.imshow(mDist, interpolation='nearest', cmap=plt.cm.ocean, extent=(0.5,10.5,0.5,10.5))
    plt.colorbar()
    plt.show()

cutDist = 0.025
totIdx = np.array(range(mDist.shape[0]))
totBool = np.array([False]*mDist.shape[0])
chainL = []

chainId = 0
spotL['chainId'] = np.nan
for i in range(mDist.shape[0]):
#    chainL.append(spotL.iloc[totIdx[mDist[i]<cutDist]][['XCenter','YCenter']])
    pNei = mDist[i]<cutDist
    if np.isnan(spotL.loc[i,'chainId']):
        spotL.loc[totIdx[pNei],['chainId']] = chainId        
        chainId += 1
    else:
        spotL.loc[totIdx[pNei],['chainId']] = spotL.loc[i,'chainId']

for i in range(mDist.shape[0]):
    pNei = mDist[i]<cutDist
    spotL.loc[totIdx[pNei],['chainId']] = spotL.loc[i,'chainId']

print(len(set(spotL['chainId'])))
# print(spotL.loc[spotL['chainId']==20,['chainId']])
# print(spotL.loc[spotL['chainId']==0,['chainId']])
        
spotM.to_csv(os.environ['LAV_DIR']+'/out/mc_whitespotNearest.csv',sep="\t")
spotL.to_csv(os.environ['LAV_DIR']+'/out/mc_whitespotCentroid.csv',sep="\t")
spotC.to_csv(os.environ['LAV_DIR']+'/out/mc_whitespotClust.csv',sep="\t")


    
    

if False:
    mHist, xHist = np.histogram(mDist1,bins=100) #look for a common minimum
    plt.plot(xHist[:-1],mHist)
    plt.show()
    last = Z[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    print("clusters:", k)

if False:
    c, coph_dists = cophenet(Z,mDist1)
    from scipy.cluster.hierarchy import inconsistent
    depth = 5
    incons = inconsistent(Z, depth)
    print(incons[-10:])
    plt.title('Hierarchical Clustering Dendrogram (truncated)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(Z,truncate_mode='lastp',  # show only the last p merged clusters
        p=12,  # show only the last p merged clusters
        #    show_leaf_counts=False,  # otherwise numbers in brackets are counts
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()




mHist, xHist = np.histogram(spotM['count'],bins=20) #look for a common minimum
plt.plot(xHist[:-1],mHist)
plt.show()

ndays = 90
t = np.linspace(0,ndays,ndays)
y1 = t
y2 = [24./15.*(x - 23) if x >= 23 else 0 for x in t]
base = datetime.datetime.strptime("2017-10-01","%Y-%m-%d")
t = [base + datetime.timedelta(days=x) for x in range(ndays)]
plt.plot(t,y1)
plt.plot(t,y2)
plt.show()


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
mat = np.matrix([[1.,.1,.6,.4],[.1,1.,.1,.2],[.6,.1,1.,.7],[.4,.2,.7,1.]])
print(SpectralClustering(2).fit_predict(mat))
eigen_values, eigen_vectors = np.linalg.eigh(mat)
print(KMeans(n_clusters=2, init='k-means++').fit_predict(eigen_vectors[:, 2:4]))
from sklearn.cluster import DBSCAN
DBSCAN(min_samples=1).fit_predict(mat)


    
