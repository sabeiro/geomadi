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
import geomadi.train_viz as t_v
import geomadi.series_stat as s_
import shapely as sh
from shapely.geometry.polygon import Polygon
from ast import literal_eval
import geomadi.geo_enrich as g_e

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

if 'dens' in locals(): del dens
if 'motL' in locals(): del motL

metric = "motion"
if(len(sys.argv) > 1): metric = sys.argv[1]
projDir = baseDir + "raw/gps/" + "motion" + "/"
dL = os.listdir(projDir)

threshold = 6.340532509207963e-06*.5
threshold = 0.0001
threshold = 0.000013#1.08e-06
print("threshold %.2f" % (threshold))

for d in dL:
    print(d)
    den = pd.read_csv(projDir + d,compression="gzip")
    den = den.replace(float('nan'),1e-10)
    den.rename(columns={"m_x":"x","m_y":"y"},inplace=True)
    dwe = den[den['m_speed'] < threshold]
    mot = den[den['m_speed'] > threshold]
    dwe = gO.dwellingBox(dwe,max_iter=1,threshold=30,isGroup=True)
    if not dwe.shape[0] == 0:
        if 'dens' not in locals(): dens = dwe
        else: dens = g_o.mergeSum(dens,dwe,cL=['octree'])
        dens = g_o.densGroupAv(dens,max_iter=0,threshold=10)
    if not mot.shape[0] == 0:
        mot = gO.motionPair(mot,precision=15)
        if 'motL' not in locals(): motL = mot
        else: motL = g_o.mergeSum(motL,mot,cL=['origin','destination'])
        motL = g_o.densOriginDest(motL,max_iter=1,threshold=10)
    
dwe = g_o.densGroup(dens,max_iter=5,threshold=30)
dwe.loc[:,'octree'] = dwe['octree'].astype(np.int64)
dwe.to_csv(baseDir + "raw/gps/"+"dwelling"+".csv.gz",compression="gzip",index=False)
poly = gO.geoDataframe(dwe)
poly.to_file(baseDir + "gis/gps/"+"dwelling"+".shp")

mot = g_o.densOriginDest(motL,max_iter=5,threshold=20)
mot.to_csv(baseDir + "raw/gps/"+"motion"+".csv.gz",compression="gzip",index=False)
poly = gO.geoLines(mot)
poly.to_file(baseDir + "gis/gps/"+"motion"+".shp")

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
