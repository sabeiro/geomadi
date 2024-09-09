import os, sys, gzip, random, csv, json, re
os.environ['LAV_DIR'] = "/home/"+os.environ['USER']+"/lav/motion/"
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import geomadi.proc_lib as plib
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")
from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Europe/Berlin')

#os.environ["SPARK_HOME"] = "/usr/hdp/current/spark-client"
conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('ttrajectory_counts')
conf.set("spark.executor.memory", "16g")
conf.set("spark.executor.cores", "16")
conf.set("spark.executor.instances", "16")

cred = json.load(open(baseDir + "credenza/geomadi.json"))
custD = "mc"
idField = "id_poi"
mapL = pd.read_csv(baseDir+"raw/"+custD+"/map_cilac.csv")
poi = pd.read_csv(baseDir+"raw/"+custD+"/poi.csv")

import importlib
importlib.reload(plib)
import shutil
print('------------------------act-cilac-------------------------------')
if False:
    idlist = list(set(mapL['cilac']))
    projDir = baseDir + "log/"+custD+"/act_cilac/"
    fL = os.listdir(projDir)
    f = fL[0]
    dfL = []
    for f in fL:
        fName = "_".join(f.split("_")[2:]).split(".")[0]
        zipD, fL = plib.parseTar(projDir,f,idlist=idlist,patterN="part-00000")
        zipD = zipD.withColumn('time',zipD.time.substr(0,19))
        zipD = zipD.where(col("time").isNotNull())
        zipD = zipD.where(col("count").isNotNull())
        if custD == "tank":
            newColumns = ["cilac","chi","time","count"]
            for i,c in enumerate(zipD.columns):
                zipD = zipD.withColumnRenamed(c,newColumns[i])
            zipD = zipD.where(zipD['chi'] >= 0)
            zipD = zipD.withColumn("chi",zipD["chi"].cast(StringType()))
            udf1 = udf(lambda x:x[:-2],StringType())
            zipD = zipD.withColumn('chi',udf1('chi'))
        zipD.coalesce(10).write.mode("overwrite").format('com.databricks.spark.csv').save(baseDir + "raw/mc/act_cilac")
        tripD = zipD.toPandas()
        dfL.append(tripD)
        shutil.rmtree(fL[0].split("/")[0])

    actD = pd.concat(dfL)
    pL = ["cilac"]
    if custD == "tank":
        pL = ["cilac","chi"]
    sact = actD.pivot_table(index=pL,columns="time",values="count",aggfunc=np.sum).fillna(0).reset_index()
    sact.loc[:,"chi"] = sact['chi'].astype(int)
    #sact.replace(float('NaN'),0,inplace=True)
    #sact.loc[:,"id_clust"] = sact[['id_poi','chi']].apply(lambda x: "%d-%d" %(x[0],x[1]), axis=1)
    sact.sort_values(["cilac"],inplace=True)
    sact.to_csv(baseDir + "raw/"+custD+"/act_cilac.csv.gz",compression="gzip",index=False)


if False:
    print('-------------------------act-chi-----------------------------')
    df, fL = plib.parsePath(baseDir + "log/tank/act_chi",is_lit=True)
    tripD = df.toPandas()
    fL1 = [x[29:31] for x in fL]
    tripD.loc[:,"dist"] = tripD['dir'].apply(lambda x: fL1[x]).values
    del tripD['dir']
    tripD.columns = ["id_zone","chi","time","count","dist"]
    tripD.loc[:,"dist"] = tripD['dist'].astype(int)
    tripD = tripD[tripD['dist'] > 0] ## wrong mapping
    tripD = tripD[tripD['time'] == tripD['time']] ## no time aggregation
    sumN = [{"name":"initial","N":sum(tripD['count'])}]
    if False:
        tripD = tripD[tripD['chi'] != '-1']
        tripD = tripD[tripD['chi'] != 'null']
        tripD.loc[:,"chi"] = tripD['chi'].astype(float)
        tripD.loc[:,"chi"] = tripD['chi'].astype(int)
        tripD = tripD[tripD['chi'] >= 0]    
    sumN.append([{"name":"no chirality","N":sum(tripD['count'])}])
    tripD.loc[:,"time"] = tripD['time'].apply(lambda x: str(x) + "00:00:00")
    tripD.loc[:,"id_zone"] = tripD['id_zone'].astype(int)
    tripD = tripD[tripD['id_zone'] >= 0]
    sumN.append([{"name":"no zone","N":sum(tripD['count'])}])
    tripD.loc[:,"id_clust"] = tripD[['id_zone','chi']].apply(lambda x: "%d-%d" % (x[0],x[1]),axis=1)
    del tripD['id_zone'], tripD['chi']
    tripD = tripD.groupby(["id_clust","time","dist"]).agg(sum).fillna(0).reset_index()
#    tripD.loc[:,"id_zone"] = act['id_zone'].apply(lambda x: x.split("-")[0])
    tripD.to_csv(baseDir + "raw/tank/act_month.csv.gz",index=False,compression="gzip")

cel = pd.read_csv(baseDir + "raw/basics/antenna_spec.csv.gz",compression="gzip")
