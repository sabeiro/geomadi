import os, sys, gzip, random, csv, json
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime
import geomadi.geo_octree as g_o

dens = pd.read_csv(baseDir + "raw/gps/dens_month.csv.gz",compression="gzip",dtype={'octree':str})
ret = gpd.read_file(baseDir + "gis/retail/retail.shp")

import importlib
importlib.reload(g_o)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])

for j,k in ret.iterrows():
    poly = k['geometry']
    g3 = gO.boundingBox(poly.exterior.bounds)
    setL = dens['octree'].apply(lambda x: x[:len(g3)] == g3)
    densP = dens[setL]
    count = 0.
    for i,g in densP.iterrows():
        p = sh.geometry.Polygon(gO.decodePoly(g['octree']))
        a = p.intersection(poly)
        share = a.area/p.area
        count += g['n']*share
    ret.loc[j,"n"] = [count]
    
ret.to_file(baseDir + "gis/retail/retail.shp")
ret[['Name','n']].to_csv(baseDir + "raw/gps/retail.csv",index=False)

if False:
    polyL = [sh.geometry.Polygon(gO.decodePoly(g3))]
    polyF = gpd.GeoDataFrame({"geometry":polyL,'n':ret['name'],'octree':[g3]})
    polyF.to_file(baseDir + "gis/gps/bbox_octree.shp")


print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
