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
from ast import literal_eval

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])

metric = "speed"
n_iter = 10
if(len(sys.argv) > 1): metric = sys.argv[1]
if(len(sys.argv) > 2): n_iter = sys.argv[2]
projDir = baseDir + "raw/gps/" + metric + "/"
dL = os.listdir(projDir)

for d in dL:
    print(d)
    den = pd.read_csv(projDir + d)
    den = den.replace(float('nan'),1e-10)
    if 'dens' not in locals(): dens = den
    else: dens = g_o.mergeSum(dens,den)
    dens = g_o.densGroupAv(dens,max_iter=0,threshold=30)

dens = g_o.densGroupAv(dens,max_iter=n_iter,threshold=60)
dens.loc[:,'octree'] = dens['octree'].astype(np.int64)

print(dens.describe())

dens.to_csv(baseDir + "raw/gps/"+metric+".csv.gz",compression="gzip",index=False)
poly = gO.geoDataframe(dens)
poly.to_file(baseDir + "gis/gps/"+metric+".shp")

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


