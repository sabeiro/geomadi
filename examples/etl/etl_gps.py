import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.lib_graph as gra
import geomadi.geo_octree as g_o
import geomadi.train_viz as t_v
import shapely as sh
from shapely.geometry.polygon import Polygon

gO = g_o.h3tree(BoundBox=[5.866,47.2704,15.0377,55.0574])
projDir = baseDir + "log/gps/2019040100/"
fL = os.listdir(projDir)
fL = [x for x in fL if bool(re.search("gz",x))]
pos = []
for f in fL:
    print(f)
    pos.append(pd.read_csv(projDir+f,compression="gzip",sep="\t",names=["ts","id","type","y","x","accuracy","tz"]))
pos = pd.concat(pos)
pos = pos.sort_values(['id','ts'])
gpi = pos.groupby("id").agg(np.mean).reset_index()
gpn = pos.groupby("id").agg(len).reset_index()
print("users %d density %.2f" % (gpi.shape[0],pos.shape[0]/gpi.shape[0]))
hist = np.unique(pos['id'],return_counts=True)

if False:
    p = pos[pos['id'] == gpn.tail(1)['id'].values[0]]
    p = p.sort_values('ts')
    t_v.plotHist(p['ts'].values)
    plt.show()
    
ts = pos['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
hour = ts.apply(lambda x: x.hour)
pos.loc[:,'hour'] = hour
if False:
    hourH = np.unique(pos['hour'],return_counts=True)
    plt.bar(hourH[0],hourH[1]/gpi.shape[0])
    plt.xlabel("hour")
    plt.ylabel("count/user")
    plt.show()

BBox = [[min(pos['x']),min(pos['y'])],[max(pos['x']),min(pos['y'])]]
BoundBox = [5.866,47.2704,15.0377,55.0574]
precDigit = 11
pos.loc[:,'octree'] = pos.apply(lambda x: gO.encode(x['x'],x['y'],precision=precDigit),axis=1)

print('-------------------trajectories-----------------')
def clampF(x):
    return pd.Series({"t":list(x['ts']),"g":list(x['octree'])})
traj = pos.groupby('id').apply(clampF).reset_index()
traj.loc[:,"n"] = traj.apply(lambda x: len(x['t']),axis=1)
traj.loc[:,"dt"] = traj['t'].apply(lambda x: [x-y for x,y in zip(x[1:],x[:-1])])
traj.loc[:,"dg"] = traj['g'].apply(lambda x: [gO.calcDisp2(x,y) for x,y in zip(x[1:],x[:-1])])
traj[['id','t','g']].to_csv(baseDir + "raw/gps/traj.csv.gz",compression="gzip",index=False)

pos.loc[:,"n"] = 1.
precDigit = 11
pos.loc[:,"octree1"] = pos['octree'].apply(lambda x: x[:precDigit])

dens.to_csv(baseDir + "raw/gps/dens.csv.gz",compression="gzip",index=False)

if False:
    import importlib
    importlib.reload(g_o)
    importlib.reload(g_m)
    gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
    gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

    dens = pd.read_csv(baseDir + "raw/gps/dens.csv.gz",compression="gzip",index_col=0)
    dens = dens.sum(axis=1).reset_index()
    dens.columns = ['octree','n']
    densD = g_o.densGroup(dens,max_iter=5,threshold=90)
    dens.to_csv(baseDir + "raw/gps/dens_month.csv.gz",compression="gzip")
    poly = gO.geoDataframe(densD)
    poly.to_file(baseDir + "gis/gps/gps_octree.shp")
    
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
