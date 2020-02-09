#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
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

with open(baseDir + '/credenza/geomadi.json') as f:
    cred = json.load(f)

with open(baseDir + '/raw/metrics.json') as f:
    metr = json.load(f)['metrics']

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
print('-----------------------near-nodes-tank-------------------')
poi = pd.read_csv(baseDir + "raw/telia/tile_list.csv")
# exist = pd.read_csv(baseDir + "raw/telia/already_established.csv")
# poi = poi[~poi['Avsni_119'].isin(exist['THS_id'])]

poi.loc[:,"tile_id"] = poi["tile_id"].apply(lambda x: int(x))
coll = client["telia_se_grid"]["grid_250"]
Nei = coll.find({'tile_id':{"$in":[int(x) for x in poi['tile_id']]}})
tileL = []
for neii in Nei:
    g = sh.geometry.Polygon(neii['geom']['coordinates'][0])
    tileL.append({'tile_id':neii['tile_id'],"x":g.centroid.xy[0][0],"y":g.centroid.xy[1][0]})
                                                            
tileL = pd.DataFrame(tileL)
if False:
    poi = pd.merge(poi,tileL,left_on="tile_id",right_on="tile_id",how="left")
    coll = client["telia_se_infra"]["nodes"]
    neiDist = 400.
    nodeL = []
    for i,poii in poi.iterrows():
        poi_coord = [x for x in poii.ix[['x','y']]]
        neiN = coll.find({'loc':{'$nearSphere':{'$geometry':{'type':"Point",'coordinates':poi_coord},'$minDistance':0,'$maxDistance':neiDist}}})
        nodeId = []
        for neii in neiN:
            nodeL.append({"tile_id":poii['tile_id'],"x_node":neii['loc']['coordinates'][0],"y_node":neii['loc']['coordinates'][1],"speed":neii['max_speed']})
            break

    nodeL = pd.DataFrame(nodeL)
    nodeL = nodeL.groupby("tile_id").first().reset_index()
    poi = pd.merge(poi,nodeL,left_on="tile_id",right_on="tile_id",how="left")

    coll = client["telia_se_grid"]["grid_250"]
    tileL = []
    for i,poii in poi.iterrows():
        poi_coord = [x for x in poii.ix[['x_node','y_node']]]
        neiN = coll.find({'geom':{'$geoIntersects':{'$geometry':{'type':"Point",'coordinates':poi_coord}}}})
        for neii in neiN:
            tileL.append({"tile_id":poii['tile_id'],"tile_id_upload":neii['tile_id']})
            break
    
    tileL = pd.DataFrame(tileL)
    
tileL = tileL.groupby("tile_id").first().reset_index()
poi = pd.merge(poi,tileL,left_on="tile_id",right_on="tile_id",how="left")
#poiUp = poi[['Avsni_119','tile_id_upload','x_node','y_node']]
poiUp = poi[['Avsni_119','tile_id','y','x']]
poiUp.loc[:,"start-date"] = "2018-01-01" ## 1514764800
poiUp.loc[:,"end-date"] = "2018-12-31" ##1546214400
poiUp.loc[:,"gridVerticalRadius"] = 1
poiUp.loc[:,"gridHorizontalRadius"] = 1
poiUp.loc[:,"comment"] = "upload after odm_test_clust"

poiUp.to_csv(baseDir + "raw/telia/poi_upload.csv",sep=";",header=None,index=False)
poi.to_csv(baseDir + "raw/telia/poi_upload_june.csv",sep=",",index=False)

