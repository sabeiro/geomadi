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
filterN = "retail"
import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.novtree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

ret = gpd.read_file(baseDir + "gis/retail/retail.shp")
ags8 = gpd.read_file(baseDir + "gis/geo/ags8.shp")
ags8.loc[:,'bbox'] = ags8['geometry'].apply(lambda x: x.convex_hull.exterior.bounds)
if False:
    print('-----------------export-bounding-box--------------------')
    importlib.reload(g_o)
    gO = g_o.quadtree(BoundBox=[5.866,47.2704,15.0377,55.0574])
    #gO = g_o.quadtree(BoundBox=[0.,0.,90.,90.])
    print(gO.nx)
    boxS = ret.copy()
    boxL = boxS['geometry'].apply(lambda x: str(gO.boundingBox(x.exterior.bounds)))
    print(boxL.head())
    boxS.loc[:,"geometry"] = boxL.apply(lambda x: sh.geometry.Polygon(gO.decodePoly(x)))
    boxS.to_file(baseDir + "gis/retail/retail_bbox.shp")

    
bbox = ags8.loc[ags8['GEN'] == 'Berlin','bbox'].iloc[0]
gBerlin = str(gO.boundingBox(bbox))
boxL = [gBerlin]
boxL = ret['geometry'].apply(lambda x: str(gO.boundingBox(x.exterior.bounds)))

projDir = baseDir + "raw/gps/traj/"
dL = os.listdir(projDir)
dL = [x for x in dL if bool(re.search("traj",x))]
d = dL[0]
def procTraj(d):
    print(d)
    meanSpeed, clusterRatio = 0., 0.
    motV, quiver, trajL = [], [], []
    traj = pd.read_csv(projDir+d,compression="gzip",dtype={"bbox":str})
    trajR = []
    for i,g in traj.iterrows():
        gra.progressBar(i,traj.shape[0],time_start)
        boxB = [x == g['bbox'][:len(x)] for x in boxL]
        if sum(boxB) == 0: continue
        retS = ret.loc[boxB]
        tx = literal_eval(g['tx'])
        p2 = [sh.geometry.Point(x[1],x[2]) for x in tx]
        isIn = False
        retV = []
        for i1,g1 in retS.iterrows():
            p1 = g1['geometry']
            setL = [p1.contains(x) for x in p2]
            if sum(setL) > 0:
                isIn = True
                t = [x[0] for x,y in zip(tx,setL) if y]
                dt = max(t) - min(t)
                retV.append({"poi":g1['id'],"event":sum(setL),"dt":dt})
        if isIn:
            t = g.copy()
            t.loc["ret_list"] = retV
            trajR.append(t.to_dict())
    trajR = pd.DataFrame(trajR)
    trajR.to_csv(baseDir + "raw/gps/"+filterN+"/traj/"+d,compression="gzip",index=False)
    return i
    
#for d in dL: procTraj(d)

pool = ThreadPool(10)
results = pool.map(procTraj,dL)
pool.close()
pool.join()

# trajL = pd.concat(results)
# pd.DataFrame(trajL).to_csv(baseDir + "raw/gps/traj_kpi.csv",index=False)

print('time per file %f min' % ( (time.time()-time_start)/len(dL)/60))

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

