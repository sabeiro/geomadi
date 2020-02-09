#%pylab inline
# -*- coding: utf-8 -*-
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from tzlocal import get_localzone
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

tz = get_localzone()
nBin = 4
gradMeter = 111122.19769899677
max_d = 5000./gradMeter

import geopandas as gpd
import shapely as sha
import shapely.speedups
import geohash
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
shapely.speedups.enable()

if False:
    plog('--------------------------reshape-popDens-----------------------')
    densG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/pop_density.shp")
    centL = densG['geometry'].apply(lambda x: x.centroid)
    densG.loc[:,"hash"] = centL.apply(lambda x: geohash.encode(x.xy[0][0],x.xy[1][0],precision=5))
    def clampF(x):
        return pd.Series({"pop_dens":x['Einwohner'].sum()
                          ,"flat_dens":x['Wohnfl_Bew'].sum()
                          ,"foreign":x['Auslaender'].sum()
                          ,"women":x['Frauen_A'].sum()
                          ,"young":x['unter18_A'].sum()
                          ,"geometry":cascaded_union(x['geometry'])
                          ,"household":x['HHGroesse_'].sum()
                          ,"n":len(x['Flaeche'])
        })
    densG = densG.groupby("hash").apply(clampF).reset_index()
    densG.loc[:,'geometry'] = densG['geometry'].apply(lambda f: f.convex_hull)
    for i in ['pop_dens','flat_dens','foreign','women','young','household']:
        densG.loc[:,i] = densG[i]/densG['n']
    densG = gpd.GeoDataFrame(densG)
    densG.to_file(baseDir + "gis/geo/pop_dens_2km.shp")

if False:
    plog('--------------------------popDens-pro-zip----------------------')
    densG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/pop_dens.shp")
    centL = gpd.GeoDataFrame(geometry=densG['geometry'].apply(lambda x: x.centroid),index=densG.index)
    zipG = gpd.GeoDataFrame.from_file(baseDir + "gis/geo/zip5.shp")
    zipG.loc[:,"dens"] = 0.
    for i,p in zipG.iterrows():
        pMask = densG.intersects(p['geometry'])
        dens = densG[pMask]
        if dens.shape[0] == 0:
            continue
        zipG.ix[i,"dens"] = zipG.ix[i,"dens"] + sum(dens["Einwohner"])
    zipG[['PLZ','dens']].to_csv(baseDir + "gis/destatis/zip_popDens.csv",index=False)


    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')


    
