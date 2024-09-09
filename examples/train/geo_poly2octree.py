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
import time
time_start = time.time()

import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

pop = gpd.read_file(baseDir + "gis/geo/pop_dens.shp")

import importlib
importlib.reload(g_o)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
for k,g in pop.iterrows():
    gra.progressBar(n,pop.shape[0],time_start)
    if (k%10) == 0: print("process %.2f\r" % (k/pop.shape[0]),end="\r",flush=True)
    oct1 = gO.intersectGeometry(g,isGeometry=True)
    if k == 0: octD = oct1
    else: octD = octD.append(oct1)
    #if k == (50-1): break

print('summing up')
octD.loc[:,"n"] = octD['Einwohner']*octD['ratio']
octD.replace(float('nan'),int(0),inplace=True)
dens = g_o.densGroupAv(octD,max_iter=0,threshold=120)
poly = gO.geoDataframe(dens)
poly.to_file(baseDir + "gis/gps/pop_dens_oct.shp")

if False:
    octD = gpd.GeoDataFrame(octD)
    octD.to_file(baseDir + "gis/gps/pop_dens_oct.shp")


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
