import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import geomadi.proc_lib as plib
import findspark
findspark.init()
import pyspark
#sc = pyspark.SparkContext('local[*]')
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

if False:
    plog('----------------------------------------zip2zip---------------------------')
    projDir = "log/nissan/odm_via/2017/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_0165"
    projDir = "log/nissan/odm_longbreak/2018/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_5810"
    projDir = "log/nissan/odm_shortbreak/2018/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_5808"
    dateL = os.listdir(baseDir + projDir)
    df = sqlContext.read.parquet(baseDir + projDir)
    df = df.drop("time_destination")
    df = df.drop("time_origin")    
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "log/nissan/odm_longbreak.csv.tar.gz",compression="gzip",index=False)

    import importlib
    importlib.reload(plib)
    projDir = "log/nissan/odm_via_zip/"
    dateL = os.listdir(baseDir + projDir)
    df = plib.parseParquet(baseDir + projDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "raw/odm_zip.csv")
    
if False:
    via = plib.readRemote("tdg/qsm/20190109_1358_odm_via_via_thuringen_new/2019/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_1197")
    print("total traj count %f" % via['count'].sum())
    via.to_csv(baseDir + "raw/nissan/via_via_v6.csv",index=False)
    
    via = plib.readRemote("/tdg/qsm/20181119_1436_odm_via_via_thuering/2018/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_0705")
    print("total traj count %f" % via['count'].sum())
    via.to_csv(baseDir + "raw/nissan/via_via_v5.csv",index=False)
    
    fileL = ["/tdg/2018/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_0082"
    ,"/tdg/2018/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_0141"
    ,"/tdg/2018/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_0139"]
    for i,vi in enumerate(fileL):
        df = sqlContext.read.parquet(cred['hdfs']['address'] + vi)
        if vi==fileL[0] :
            traD = df
        else :
            traD = traD.unionAll(df)
    via = traD.toPandas()
    via.to_csv(baseDir + "raw/nissan/via_via_v4.csv",index=False)
        
if False: ##------------------------nissan----------------------------------------------------
    projDir = "log/nissan/odm_via/2017/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_0165"
    projDir = "log/nissan/odm_longbreak/2018/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_5810"
    projDir = "log/nissan/odm_shortbreak/2018/custom_aggregations/odm_result/custom_odm_result_application_1513256541702_5808"
    dateL = os.listdir(baseDir + projDir)
    df = sqlContext.read.parquet(baseDir + projDir)
    df = df.drop("time_destination")
    df = df.drop("time_origin")    
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "log/nissan/odm_longbreak.csv.tar.gz",compression="gzip",index=False)

    import importlib
    importlib.reload(plib)
    projDir = "log/nissan/odm_via_zip/"
    dateL = os.listdir(baseDir + projDir)
    df = plib.parseParquet(baseDir + projDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "raw/odm_zip.csv")
    
