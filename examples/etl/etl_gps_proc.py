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
import geomadi.train_viz as t_v
import shapely as sh
from shapely.geometry.polygon import Polygon
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf, struct
sqlContext = SQLContext(sc)

projDir = baseDir + "log/gps/"
dL = os.listdir(projDir)
BoundBox = [5.866,47.2704,15.0377,55.0574]

precDigit = 13
def get_octree(x,y):
    return g_o.encode(x,y,precision=precDigit,BoundBox=BoundBox)

udf_get_octree = udf(get_octree,StringType())

for d in dL:
    print(d)
    fL = os.listdir(projDir+d+"/")
    fL = [x for x in fL if bool(re.search("gz",x))]
    pos = []
    for f in fL:
        df = sqlContext.read.format('com.databricks.spark.csv').option("header","false").option("inferSchema","true").option("delimiter",'\t').load(projDir+d+"/"+f)
        newCol = ['ts','id','type','y','x','accuracy','tz']
        for i,j in zip(df.columns,newCol): df = df.withColumnRenamed(i,j)
        if f==fL[0] : df = df
        else : df = df.unionAll(df)
        #pos.append(pd.read_csv(projDir+d+"/"+f,compression="gzip",sep="\t",names=["ts","id","type","y","x","accuracy","tz"]))
    ##pos = pd.concat(pos)
    df = df.withColumn("octree",udf_get_octree(struct('x','y')))
    pos = df.toPandas()
    pos = pos.sort_values(['id','ts'])
    pos.loc[:,'octree'] = pos.apply(lambda x: g_o.encode(x['x'],x['y'],precision=precDigit,BoundBox=BoundBox),axis=1)
    def clampF(x):
        return pd.Series({"t":list(x['ts']),"g":list(x['octree'])})
    traj = pos.groupby('id').apply(clampF).reset_index()
    traj.to_csv(baseDir + "raw/gps/traj_"+d+".csv.gz",compression="gzip",index=False)

    
    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
