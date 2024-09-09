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
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql import functions as func
sqlContext = SQLContext(sc)

def plog(text):
    print(text)

key_file = baseDir + '/credenza/geomadi.json'
cred = []
with open(key_file) as f:
    cred = json.load(f)

projDir = "/tdg/qsm/20190125_1425_activities_unibiail_cilac_hourly"
projDir = "/tdg/2018/11/06/QA/11.6/31.4M/trips/activity"

projDir = cred['hdfs']['address'] + projDir

df = sqlContext.read.parquet(projDir)
df.show()
df = df.withColumn("tac",df.other_fields['tac'])

for dd in df.take(20):
    print(dd.tac)


def count_not_null(c, nan_as_null=False):
    pred = col(c).isNotNull() & (~func.isnan(c) if nan_as_null else func.lit(True))
    return func.sum(pred.cast("integer")).alias(c)

df.agg(count_not_null('tac')).show()
ddf = df.where(df['tac'] > 0)
n1 = df.count()
n2 = df.where(df['tac'] > 0).count()
print(n2/n1)
0.7144750729591113

df.agg(*[count_not_null(c) for c in df.columns]).show()
df.agg(*[count_not_null(c, True) for c in ['tac']]).show()



print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
