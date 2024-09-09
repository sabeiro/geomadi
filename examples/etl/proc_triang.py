import os, sys, gzip, random, csv, json
sys.path.append(os.environ['LAV_DIR']+'/src/py/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
# import findspark
# findspark.init()
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
import geomadi.geo_enrich as g_e
from pyspark.sql import functions as func
sqlContext = SQLContext(sc)
import pymongo
import osmnx as ox

def plog(text):
    print(text)

cred = json.load(open(baseDir + '/credenza/geomadi.json'))

conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('ViaVia Check')
conf.set("spark.executor.memory", "8g")
conf.set("spark.executor.cores", "8")
conf.set("spark.executor.instances", "8")
    
projDir = "/tdg/qsm/20190125_1425_activities_unibiail_cilac_hourly"
projDir = "/tdg/ignored/2018/11/06/QA/11.6/31.4M/trips/activity"
projDir = "/tdg/2018/11/06/aggregated_events_munich_filtered"
projDir = cred['hdfs']['address'] + projDir
projDir = 'hdfs://172.25.100.51:8020/' + projDir
custD = "mc"
idField = "id_poi"

#df = sqlContext.read.parquet(projDir+"_metadata",projDir+"_common_metadata",projDir+"/part-r-00001*")
df = sqlContext.read.parquet(projDir)
df.show()

plog('-------------------isochrone-----------------')
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
mapL = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
mapL = mapL[mapL[idField] == poi[idField][0]]
lacS = np.unique([x.split("-")[1] for x in mapL['cilac']])
ciS = np.unique([x.split("-")[0] for x in mapL['cilac']])

dx, dy = 0.05, 0.025
client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
collE = client["tdg_infra"]["infrastructure"]
import importlib
importlib.reload(g_e)
polyL = []
g = poi[poi[idField] == np.unique(poi[idField])[0]]
xc, yc = g['x'].values[0], g['y'].values[0]
geoD = g_e.localPolygon(xc,yc,dx,dy,collE)

response = collE.find({"cell_ci":int(cilac.split("-")[0]),"cell_lac":int(cilac.split("-")[1])})
lista = (response[0]["centroid"])

ox.plot_graph(G)
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
