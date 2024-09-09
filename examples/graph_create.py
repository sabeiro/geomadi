import os, sys, gzip, random, csv, json, datetime, re
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
import geomadi.geo_octree as g_o
#import lernia.train_viz as t_v
import albio.series_stat as s_s
import albio.series_interp as s_i
import shapely as sh
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import shapely.speedups
shapely.speedups.enable()
import geomadi.geo_octree as g_o
import geomadi.graph_ops as g_p
from h3 import h3
from rtree import index

if False:
    """download Berlin graph"""
    berG = ox.graph_from_place("Berlin")
    # berG = g_p.downloadGraph("Berlin")
    g_p.save(berG,baseDir + "/log/geo/berlin")
    

if False:
  """load graph"""
  street = gpd.read_file(baseDir + "log/geo/berlin_street.shp")
  G = g_p.line2graph(street.head(50))

    
import importlib
importlib.reload(g_p)

vanG = g_p.removeType(berG,move_type="driver")
cycG = g_p.removeType(berG,move_type="cycle")
g_p.save(vanG,baseDir + "/log/geo/berlin_van")
g_p.save(cycG,baseDir + "/log/geo/berlin_cycle")

if False:
  """pivot on octrees on different resolutions"""
  vanP = gpd.read_file(baseDir + "/log/geo/berlin_van.gpkg")
  cycP = gpd.read_file(baseDir + "/log/geo/berlin_cycle.gpkg")
  gO = g_o.h3tree()
  vanP.loc[:,'octree8'] = vanP.apply(lambda x: gO.encode(x['x'], x['y'], precision=8),axis=1)
  vanP.loc[:,'octree9'] = vanP.apply(lambda x: gO.encode(x['x'], x['y'], precision=9),axis=1)
  van8 = vanP.groupby('octree8').head(1)
  van8.to_file(baseDir + "/log/geo/berlin_van8.gpkg")
  van9 = vanP.groupby('octree9').head(1)
  van9.to_file(baseDir + "/log/geo/berlin_van9.gpkg")
  cycP.loc[:,'octree8'] = cycP.apply(lambda x: gO.encode(x['x'], x['y'], precision=8),axis=1)
  cycP.loc[:,'octree9'] = cycP.apply(lambda x: gO.encode(x['x'], x['y'], precision=9),axis=1)
  cyc8 = cycP.groupby('octree8').head(1)
  cyc8.to_file(baseDir + "/log/geo/berlin_cycle8.gpkg")
  cyc9 = cycP.groupby('octree9').head(1)
  cyc9.to_file(baseDir + "/log/geo/berlin_cycle9.gpkg")
  
