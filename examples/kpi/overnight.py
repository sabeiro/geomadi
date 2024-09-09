import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import geomadi.geo_enrich as g_e

cent = pd.read_csv(baseDir + "raw/basics/centroids.csv.gz")
zipL = g_e.addRegion(cent,baseDir + "gis/geo/zip5.shp",field="PLZ")
cent.loc[:,"zip5"] = zipL
cent.to_csv(baseDir + "tmp/centroids_zip.csv.gz",compression="gzip",index=False)
