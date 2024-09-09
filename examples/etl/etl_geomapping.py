#%pylab inline
import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import geomadi.lib_graph as gra
#gra.style()

def plog(text):
    print(text)

if False:
    mcc = pd.read_csv(baseDir + "raw/basics/mcc.csv")
    nat = gpd.read_file(baseDir + "gis/geo/countries.shp")
    nat = nat.groupby("name").first().reset_index()
    nat = pd.merge(nat,mcc,left_on="name",right_on="country",how="left")
    nat = gpd.GeoDataFrame(nat)
    nat.to_file(baseDir + "gis/geo/countries.shp")



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
