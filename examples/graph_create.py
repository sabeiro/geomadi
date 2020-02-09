import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
sys.path.append('/usr/local/lib/python3.6/dist-packages/')
baseDir = os.environ['LAV_DIR']
import geomadi.geo_octree as g_o
import lernia.train_viz as t_v
import albio.series_stat as s_s
import albio.series_interp as s_i
import shapely as sh
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
import shapely.speedups
shapely.speedups.enable()
import geomadi.geo_octree as g_o
from h3 import h3
from rtree import index
import geomadi.graph_ops as g_p


street = gpd.read_file(baseDir + "gis/geo/berlin_" + "street" + ".shp")

import importlib
importlib.reload(g_p)
G = g_p.line2graph(street.head(50))
sim_G = g_p.simplifyGraph(G)




