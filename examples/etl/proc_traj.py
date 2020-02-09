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
import geomadi.geo_motion as g_m
import lernia.train_viz as t_v
import albio.series_stat as s_s
import shapely as sh
from shapely.geometry.polygon import Polygon
from ast import literal_eval
from multiprocessing.dummy import Pool as ThreadPool
import lernia.multi_thread as m_t
import time
time_start = time.time()

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

ags8 = gpd.read_file(baseDir + "gis/geo/ags8.shp")
ags8.loc[:,'bbox'] = ags8['geometry'].apply(lambda x: x.convex_hull.exterior.bounds)
bbox = ags8.loc[ags8['GEN'] == 'Berlin','bbox'].iloc[0]
gBerlin = str(gO.boundingBox(bbox))

projDir = baseDir + "raw/gps/traj/"
dL = os.listdir(projDir)
dL = [x for x in dL if bool(re.search("traj",x))]
threshold = 0.0002
threshold = 0.000013#1.08e-06
print('threshold %f' % threshold)
def procTraj(d):
    print(d)
    meanSpeed, clusterRatio = 0., 0.
    motV, quiver, trajL = [], [], []
    traj = pd.read_csv(projDir+d,compression="gzip",converters={"tx":literal_eval})
    traj = traj[traj['n'] > 30]
    traj = traj[ traj['bbox'].apply(lambda x: str(x)[:len(gBerlin)] == gBerlin) ]
    traj = traj.sort_values('n',ascending=False).reset_index()
    for i,g in traj.iterrows():
        X = np.array(g['tx'])
        XV = gM.motion(X,isSmooth=True,steps=10)
        mot, clustL = gM.cluster(XV,threshold=threshold)
        speed = gO.motion(XV,precision=15)
        trajL.append({"cluster_ratio":mot.shape[0]/X.shape[0]})
        meanSpeed = .5*(meanSpeed + np.mean(speed['speed']))
        clusterRatio = .5*(clusterRatio + mot.shape[0]/X.shape[0])
        motV.append(mot)
        quiver.append(speed)
        if (i%30) == 0:
            print("%s-%.2f) current point %04d speed %f cluster ratio %.2f" % (d.split(".")[0],i/traj.shape[0],X.shape[0],meanSpeed,clusterRatio))
        # trip = pd.DataFrame(X,columns=['tx','x','y'])
        # trip.loc[:,"c"] = clustL
        # trip.loc[:,"id"] = g['id']
    motL = pd.concat(motV)
    quiver = pd.concat(quiver)
    fName = d.split("_")[1]
    motL.to_csv(baseDir + "raw/gps/motion/motion_"+fName,compression="gzip",index=False)
    quiver.loc[:,"n"] = 1
    quiver.loc[:,"octree"] = quiver['octree'].apply(lambda x: int(str(x).split(".")[0]))
    quiver = g_o.densGroup(quiver,max_iter=3,threshold=10)
    quiver.to_csv(baseDir + "raw/gps/speed/speed_"+fName,compression="gzip",index=False)
    return pd.DataFrame(trajL)

#for d in dL: procTraj(d)

pool = ThreadPool(10)
results = pool.map(procTraj,dL)
pool.close()
pool.join()

trajL = pd.concat(results)
pd.DataFrame(trajL).to_csv(baseDir + "raw/gps/traj_kpi.csv",index=False)

print('time per file %f min' % ( (time.time()-time_start)/len(dL)/60))

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
