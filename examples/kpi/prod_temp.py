import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import pymongo
import geomadi.multi_thread as m_t
import shapely as sh
from shapely import geometry
from multiprocessing.dummy import Pool as ThreadPool
import time

idField = "id_poi"
cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))
poi = pd.read_csv(baseDir + "raw/mc/whitelist/poi.csv")

nList = 400
nPool = 100

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

time_start = time.time()

import importlib
importlib.reload(m_t)

if False:
    print('-------------------tile-2-coord-----------------')
    thread = m_t.find_coord(poi,idField,nList=nList,nPool=nPool)
    thread.setCollection(client["tdg_infra_internal"]["grid_250"])
    thread.test()
    tileL = thread.run()
    tileL.to_csv(baseDir + "out/mc/whitelist/tile_pos.csv.gz",compression="gzip",index=False)

if False:
    print('----------------------node-2-tile----------------')
    thread = m_t.node_tile(poi,idField,nList=nList,nPool=nPool)
    thread.setCollection(client["tdg_infra_internal"]["node_tile"])
    thread.test()
    tileL = thread.run()
    tileL.to_csv(baseDir + "out/mc/whitelist/tile_node.csv.gz",compression="gzip",index=False)

if False:
    print('----------------------node-2-class----------------')
    tileN = pd.read_csv(baseDir + "raw/mc/whitelist/tile_node.csv.gz",compression="gzip")
    thread = m_t.node_class(tileN[:40],idField="id_node",nList=nList,nPool=nPool)
    thread.setCollection(client["tdg_infra_internal"]["segments_col"])
    thread.test()
    tileL = thread.run()
    tileL.to_csv(baseDir + "out/mc/whitelist/node_class.csv.gz",compression="gzip",index=False)

if True:
    print('---------------------------calculate-distance-from-motorway------------------')
    tileN = pd.read_csv(baseDir + "raw/mc/whitelist/node_class.csv.gz",compression="gzip")
    motG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/motorway.shp")
    motG = motG[motG['geometry'].apply(lambda x: x.is_valid).values]
    line = motG.geometry.unary_union
    print('--------------------load-node-list-and-motorway---------------------------')
    thread = m_t.motorway_distance(tileN,idField="id_node",nList=nList,nPool=nPool)
    thread.setCollection(line)
    thread.test()
    tileL = thread.run()
    tileL.to_csv(baseDir + "out/mc/whitelist/node_motorway.csv.gz",compression="gzip",index=False)

