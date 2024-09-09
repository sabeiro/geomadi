import sys, csv, json, time
from datetime import datetime
from ast import literal_eval
import pandas as pd
import numpy as np
from pyspark.sql import SQLContext, HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import mean, min, max, sum
from pyspark.sql import functions as func
from pyspark import *
from pyspark import SQLContext
from pyspark.sql.functions import *

import glob
from pyspark.sql.functions import explode
import pytz
from collections import Counter

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import os
os.environ["SPARK_HOME"] = "/usr/hdp/current/spark-client"
conf = (SparkConf()
    .setMaster("yarn-client")
    .setAppName("first/last signal - time")
    .set("spark.deploy-mode", "cluster"))
conf.set("spark.executor.memory", "10g")
conf.set("spark.executor.cores", "10")
conf.set("spark.executor.instances", "2")
conf.set("spark.driver.maxResultSize", "10g")
conf.set("spark.driver.memory", "10g")

sc = SparkContext.getOrCreate()
sqlc = SQLContext(sc)
sc.setLogLevel("ERROR")

aggEv = "/tdg/2017/04/11/aggregated_events/"
df = sqlc.read.parquet(aggEv)
#df.show(5)

def get_last(steps): #last element of an array
    return steps[-1]

udf_get_val = udf(get_last,IntegerType())
df = df.withColumn("ts_end",udf_get_val("timestamp_arr"))
df = df.withColumn("cell_end",udf_get_val("cell_id_arr"))
df = df.withColumn("area_end",udf_get_val("area_id_arr"))

def get_first(steps): #last element of an array
    return steps[0]

udf_get_val = udf(get_first,IntegerType())
df = df.withColumn("ts_start",udf_get_val("timestamp_arr"))
df = df.withColumn("cell_start",udf_get_val("cell_id_arr"))
df = df.withColumn("area_start",udf_get_val("area_id_arr"))

udf1 = udf(lambda x:x,StringType()) ## format to ci-lac
df = df.withColumn('cell_end',udf1('cell_end'))
df = df.withColumn('area_end',udf1('area_end'))
df = df.withColumn('cell_start',udf1('cell_start'))
df = df.withColumn('area_start',udf1('area_start'))
df = df.withColumn('cilac_end',func.concat(func.col('cell_end'),func.lit('-'),func.col('area_end')))
df = df.withColumn('cilac_start',func.concat(func.col('cell_start'),func.lit('-'),func.col('area_start')))
df = df.withColumn('time_end',(df.ts_end - (df.ts_end % 3600)) - 1491861600)
df = df.withColumn('time_start',(df.ts_start - (df.ts_start % 3600)) - 1491861600)

df_sel = df.select("cilac_start", "time_start","cilac_end","time_end") ## aggregate
df_sel = df_sel.withColumn('count',func.lit(1))
df_end = df_sel.groupby(["cilac_end","time_end"]).sum('count')
df_start = df_sel.groupby(["cilac_start","time_start"]).sum('count')

#filterL = pd.read_csv("airport_filter.csv") ## only airport cilacs
#filterL.loc[:,'cilac'] = filterL.apply(lambda x: "%d-%d" % (x['CELL_CI'],x['CELL_LAC']),axis=1)
#df_sel = df_sel.where(df_sel.cilac.isin(list(filterL['cilac'])))

df_sel.show(5)

#df_end.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save("last_cilac")
#df_start.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save("first_cilac")
df_end.toPandas().to_csv("last_cilac.csv.tar.gz",index=False,compression="gzip")
df_start.toPandas().to_csv("first_cilac.csv.tar.gz",index=False,compression="gzip")
