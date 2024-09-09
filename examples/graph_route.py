import os, sys, gzip, random, csv, json, datetime, re
import osmnx as ox
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shapely as sh
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
baseDir = os.environ['LAV_DIR']
import geomadi.geo_octree as g_o
import geomadi.graph_ops as g_p
import geomadi.routing_node2node as g_r
import importlib

van8 = gpd.read_file(baseDir + "/log/geo/berlin/berlin_van8.gpkg")
nodeL = van8[['osmid','y','x','street_count','ref','highway','octree8']]
importlib.reload(g_p)
vanG = g_p.load(baseDir + "/log/geo/berlin/"+"berlin_van.gpkg")
importlib.reload(g_r)
routeL = g_r.routeList(vanG,nodeL)
routeL.to_csv(baseDir + "log/geo/berlin/oct2oct8_van.csv.gz",index=False,compression="gzip")

routeD = routeL.groupby('octree8').agg(np.sum).reset_index()
gO = g_o.h3tree()
geom = routeD['octree8'].apply(lambda x: sh.Polygon(gO.decodePoly(x)))
routeD = gpd.GeoDataFrame(routeD,geometry=geom)
routeD.to_file(baseDir + "/log/geo/berlin/berlin_oct8.gpkg")

van8 = gpd.read_file(baseDir + "/log/geo/berlin/berlin_cycle8.gpkg")
nodeL = van8[['osmid','y','x','street_count','ref','highway','octree8']]
importlib.reload(g_p)
vanG = g_p.load(baseDir + "/log/geo/berlin/"+"berlin_cycle.gpkg")
importlib.reload(g_r)
routeL = g_r.routeList(vanG,nodeL)
routeL.to_csv(baseDir + "log/geo/berlin/oct2oct8_cyc.csv.gz",index=False,compression="gzip")

if False:
  import routingpy as rp
  import numpy as np
  api_key = # get a free key at https://www.graphhopper.com/
  api = rp.Graphhopper(api_key=api_key)
  matrix = api.matrix(locations=coordinates, profile='car')
  durations = np.matrix(matrix.durations)
  print(durations)


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
  
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
