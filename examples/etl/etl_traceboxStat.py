import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
import geomadi.geo_octree as octree
import shapely as sh
import geopandas as gpd

sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
cred = json.load(open(baseDir + "credenza/geomadi.json"))

poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.loc[:,"octree"] = poi['octree'].apply(lambda x:str(x)[:9])

traX = pd.read_csv(baseDir + "raw/basics/tracebox_call_example.csv")
traX = traX[["S_TMSI","POS_FIRST_TILE","POS_LAST_TILE","POS_FIRST_LOC","POS_LAST_LOC","POS_FIRST_LON","POS_FIRST_LAT","START_TIME","POS_LAST_LON","POS_LAST_LAT","END_TIME","CALL_TYPE","CALL_STATUS","MOVING"]]

traX = traX[["S_TMSI","POS_FIRST_LON","POS_FIRST_LAT","START_TIME","POS_LAST_LON","POS_LAST_LAT","END_TIME","CALL_TYPE","CALL_STATUS","MOVING","POS_FIRST_LOC","POS_LAST_LOC","POS_FIRST_TILE","POS_LAST_TILE"]]
traX.loc[:,"dx"] = traX["POS_FIRST_LAT"] - traX["POS_FIRST_LAT"]
traX.loc[:,"dy"] = traX["POS_FIRST_LON"] - traX["POS_FIRST_LON"]
traX.loc[:,"dr"] = traX.loc[:,"dx"] - traX.loc[:,"dy"]
traX.loc[:,"dif"] = traX['POS_FIRST_TILE'] - traX['POS_LAST_TILE']
traX.loc[:,"dif1"] = traX['POS_FIRST_LOC'] - traX['POS_LAST_LOC']
print(traX[traX['MOVING'] == 1].head())
print(traX[traX['dif1'].abs() > 0].shape[0]/traX.shape[0])
print(traX[traX['dif'].abs() > 0].shape[0]/traX.shape[0])
print(traX[traX['dr'].abs() > 0].shape[0]/traX.shape[0])


plt.hist(traX['dif1'],bins=20)
plt.show()

tlib.plotHist(traX['dif1'],nBin=20,isLog=False)
tlib.plotHist(traX['vel'],nBin=20,isLog=False)

# projDir = cred['hdfs']['address'] + "/tdg/data_dumps/tracebox_2nd/parquets/CALL_20180628/part-r-00000-8f205d6e-7c22-4b60-96bc-ab4c45bfa6f7.gz.parquet"
#projDir = cred['hdfs']['address'] + "/home/hdfs/paras/tank_rast/join_poi_buf_250_generic_events"
#projDir = cred['hdfs']['address'] + "/tdg/data_dumps/tracebox_2nd/parquets/HO_20180628/part-r-00000-3745e99e-3603-43c5-816e-6df9d272de09.gz.parquet"


projDir = cred['hdfs']['address'] + "/home/hdfs/paras/unibail/generic_events/part-r-00000-084e5a0d-4dce-4a57-b807-2442c51d26d9.gz.parquet"

traD = traD.select("S_TMSI","POS_FIRST_LON","POS_FIRST_LAT","START_TIME","POS_LAST_LON","POS_LAST_LAT","END_TIME","CALL_TYPE","CALL_STATUS","MOVING","POS_FIRST_LOC","POS_LAST_LOC","POS_FIRST_TILE","POS_LAST_TILE")

if False:
    traD = traD.withColumn("dx",traD["POS_FIRST_LAT"] - traD["POS_FIRST_LAT"])
    traD = traD.withColumn("dy",traD["POS_FIRST_LON"] - traD["POS_FIRST_LON"])
    traD = traD.withColumn("dr",traD["dx"] - traD["dy"])
    traD = traD.withColumn("dif",traD["POS_FIRST_TILE"] - traD["POS_LAST_TILE"])
    traD = traD.withColumn("dif1",traD["POS_FIRST_LOC"] - traD["POS_LAST_LOC"])
    
    def not_null(x):
        if (x != x) | (x == None):
            return False
        return abs(x) > 0
    udf_not = udf(not_null, BooleanType())
    Npos = traD.where(udf_not(col("dr"))).count()
    Ndif = traD.where(udf_not(col("dif"))).count()
    Ndif1 = traD.where(udf_not(col("dif1"))).count()
    N = traD.count()
    print("%.2f %.2f %.2f" % (Npos/N,Ndif/N,Ndif1/N) )
