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

metric = "speed"
if(len(sys.argv) > 1): metric = sys.argv[1]
projDir = baseDir + "raw/gps/" + metric + "/"
dL = os.listdir(projDir)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])

for d in dL:
    print(d)
    den = pd.read_csv(projDir + d)
    if 'dens' not in locals(): dens = den
    else:
        dens = dens.merge(den,on="octree",how="outer")
        dens = dens.replace(float('nan'),0.)
        tLx = [x for x in dens.columns if bool(re.search("_x",x))]
        tLy = [x for x in dens.columns if bool(re.search("_y",x))]
        dens.loc[:,tLx] = dens[tLx].values + dens[tLy].values
        for i in tLy: del dens[i]
        dens.columns = ['octree'] + [x.split("_")[0] for x in tLx]
        
dens = g_o.densGroup(dens,max_iter=10,threshold=30)
dens.loc[:,'octree'] = dens['octree'].astype(np.int64)

dens.to_csv(baseDir + "raw/gps/"+metric+".csv.gz",compression="gzip",index=False)
poly = gO.geoDataframe(dens)
poly.to_file(baseDir + "gis/gps/"+metric+".shp")

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
