#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.geo_enrich as gen

from tzlocal import get_localzone
tz = get_localzone()
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import geopandas as gpd
def plog(text):
    print(text)

import pymongo
from scipy.spatial import cKDTree
from scipy import inf
import shapely as sh
import geomadi.train_shapeLib as shl
import geomadi.geo_octree as octree
import shapely as sh
import shapely.speedups

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))['metrics']

custD = "tank"
idField = "id_clust"
custD = "mc"
idField = "id_poi"

if False:
    plog('----------------calculate-cells-close-to-motorway----------------------------')
    lineS = [sh.geometry.LineString([x,y]) for x,y in zip(nodeG[['x1','y1']].values,nodeG[['x2','y2']].values)]
    nodeG = gpd.GeoDataFrame(lineS,columns=["geometry"])
    with open(baseDir + 'gis/"+custD+"/segments.geojson', 'w') as f:
        f.write(nodeG.to_json())
        
if False:
    plog('-----------------intersection-bse-motorway--------------------')
    coll = client["tdg_infra"]["infrastructure"]
    motG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/motorway_area.shp")
    cellL = []
    for g in np.array(motG['geometry'][0]):
        c = g.exterior.coords.xy
        c1 = [[x,y] for x,y in zip(c[0],c[1])]
        neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Polygon",'coordinates':[c1]}}}})
        neii = neiN[0]
        for neii in neiN:
            cellL.append({"cilac":str(neii['cell_ci']) + '-' + str(neii['cell_lac'])})
    cellL = pd.DataFrame(cellL)

if False:
    plog('------------------add-centroids------------------------')
    celD = pd.read_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip")
    celD = celD[['cilac','id_node', 'X', 'Y', 'north', 'height_dem', 'height', 'tilt_el','tilt_me', 'eirp', 'structure', 'system']]
    cilD = pd.read_csv(baseDir+"raw/basics/centroids.csv.gz",compression='gzip')
#    cilD = pd.read_csv(baseDir+"raw/basics/centroids.csv.gz",compression='gzip')
    celD = pd.merge(celD,cilD,on="cilac",how="left")
    setL = celD['X'] != celD['X']
    print("missing %f " % (sum(setL)/celD.shape[0]))
    celD.loc[setL,"X"] = celD['x']
    celD.loc[setL,"Y"] = celD['y']
    celD.to_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip",index=False)
    
if False:
    plog('-----------------buid-id-------------------')
    celD = pd.read_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip")
    celD = celD[celD['ci'] == celD['ci']]
    celD = celD[celD['lac'] == celD['lac']]
    celD.loc[:,"cilac"] = celD.apply(lambda x: "%d-%d" % (int(x['ci']),int(x['lac'])),axis=1)
    del celD['ci'], celD['lac']
    celD.to_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip",index=False)

