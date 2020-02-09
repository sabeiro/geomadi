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
import lernia.train_viz as t_v
import shapely as sh
from shapely.geometry.polygon import Polygon
from multiprocessing.dummy import Pool as ThreadPool
import time
time_start = time.time()

# import pyspark
# sc = pyspark.SparkContext.getOrCreate()
# from pyspark.sql import SQLContext
# from pyspark.sql.types import *
# from pyspark.sql.functions import udf
# import matplotlib.pyplot as plt
# from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
# from pyspark.sql.functions import date_format
# from pyspark.sql import functions as func
# from pyspark.sql.functions import col
# from pyspark.sql.types import IntegerType
# from pyspark.sql.functions import udf, struct
# sqlContext = SQLContext(sc)
# udf_get_octree = udf(get_octree,StringType())

projDir = baseDir + "log/gps/"
dL = os.listdir(projDir)
gO = g_o.h3tree(BoundBox=[5.866,47.2704,15.0377,55.0574])

precDigit = 11
def get_octree(x,y): return gO.encode(x,y,precision=precDigit)

def procDens(d):   
    print(d)
    fL = os.listdir(projDir+d+"/")
    fL = [x for x in fL if bool(re.search("gz",x))]
    pos = []
    for f in fL:
        # df = sqlContext.read.format('com.databricks.spark.csv').option("header","false").option("inferSchema","true").option("delimiter",'\t').load(projDir+d+"/"+f)
        # newCol = ['ts','id','type','y','x','accuracy','tz']
        # for i,j in zip(df.columns,newCol): df = df.withColumnRenamed(i,j)
        # if f==fL[0] : df = df
        # else : df = df.unionAll(df)
        pos.append(pd.read_csv(projDir+d+"/"+f,compression="gzip",sep="\t",names=["ts","id","type","y","x","accuracy","tz"]))
    pos = pd.concat(pos)
    # x,y = pos.iloc[0][['x','y']]
    # precDigit = gO.getPrecision(x,y)
    # df  = df.withColumn("octree",udf_get_octree(struct('x','y')))
    # pos = df.toPandas()
    pos = pos.sort_values(['id','ts'])
    pos.loc[:,'octree'] = pos.apply(lambda x: gO.encode(x['x'],x['y'],precision=precDigit),axis=1)
    pos.loc[:,'n'] = 1.
    freq = pos[['id','ts']].groupby('id').agg(len).reset_index()
    pos.loc[:,"ts"] = pos['ts'] - min(pos['ts'])
    dens = g_o.densGroup(pos[['octree','n','x','y','ts']],max_iter=5,threshold=30)
    dens.loc[:,'octree'] = dens['octree']#.astype(np.int64)
    def clampF(x):
        return pd.Series({
            "tx":[[x,y,z] for x,y,z in zip(x['ts'],x['x'],x['y'])]
            ,"n":len(x['ts'])
            # ,'type':x['type'].head(1)
            #,"bbox":gO.trajBBox(x['octree'].values)
            ,"bound":[min(x['x']),min(x['y']),max(x['x']),max(x['y'])]
            ,"dt":gO.trajBBox(x['ts'].values)})
    traj = pos.groupby('id').apply(clampF).reset_index()
    traj.to_csv(baseDir + "raw/gps/traj/traj_"+d+".csv.gz",compression="gzip",index=False)
    freq.to_csv(baseDir + "raw/gps/freq/freq_"+d+".csv.gz",compression="gzip",index=False)
    dens.to_csv(baseDir + "raw/gps/dens/dens_"+d+".csv.gz",compression="gzip",index=False)

pool = ThreadPool(10)
results = pool.map(procDens,dL)
pool.close()
pool.join()

print('time per file %f min' % ( (time.time()-time_start)/len(dL)/60))

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
