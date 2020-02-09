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
import geomadi.geo_enrich as g_e
import lernia.train_viz as t_v
import albio.series_stat as s_s
import shapely as sh
from shapely.geometry.polygon import Polygon
from ast import literal_eval
import pymongo
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

cred = json.load(open(baseDir + "credenza/geomadi.json"))
metr = json.load(open(baseDir + "raw/basics/metrics.json"))['metrics']

client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'],username=cred['mongo']['user'],password=cred['mongo']['pass'])

BoundBox = [5.866,47.2704,15.0377,55.0574]
idField = "id"
import importlib
importlib.reload(g_o)
importlib.reload(g_m)
gO = g_o.octree(BoundBox=[5.866,47.2704,15.0377,55.0574])
gM = g_m.motion(BoundBox=[5.866,47.2704,15.0377,55.0574])

if False:
    print('---------------------geojson-export-------------------------')
    dens = pd.read_csv(baseDir + "raw/gps/dens.csv.gz",compression="gzip",dtype={'octree':str})
    n = len(gBerlin)
    setL = dens['octree'].apply(lambda x: x[:n] == gBerlin)
    densD = dens[setL]
    poly = gO.geoDataframe(densD)
    poly.rename(columns={"n":"z"},inplace=True)
    with open(baseDir + "gis/gps/dens_berlin.geojson", 'w') as f: f.write(poly.to_json())

    dens = pd.read_csv(baseDir + "raw/gps/motion.csv.gz",compression="gzip",dtype={'octree':str})
    poly = gO.geoDataframe(densD)
    poly.rename(columns={"n":"z"},inplace=True)
    with open(baseDir + "gis/gps/dens_berlin.geojson", 'w') as f: f.write(poly.to_json())

if False:
    print('------------------------------------------------------------')
    projDir = baseDir + "raw/gps/" + "motion" + "/"
    dL = os.listdir(projDir)
    d = dL[0]
    mot = pd.read_csv(projDir + d,compression="gzip")
    tL = ['m_speed','m_angle','m_chirality','x1','y1','t1','x2','y2','t2','sr']
    json.dump(geoD,open(baseDir + "gis/gps/mot_berlin.json","w"),ensure_ascii=False)
    
    mot1 = mot[tL]
    mot1.to_csv(baseDir + "gis/gps/mot_berlin.csv",index=False)

    
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
