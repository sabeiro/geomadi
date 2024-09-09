import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import spark.proc_lib as plib
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


if False:#-----------------------demo-data--------------------------
    idL = pd.read_csv(baseDir + "raw/sel_act.csv")
    idL = idL.loc[idL['customer']=="activ"]
    projDir = baseDir + "log/samba/Customer_Projects_DE/Destatis/log_no_filter/"
    dateL = os.listdir(projDir)
    for d in dateL:
        act = pd.read_csv(baseDir + projDir + d)
        act.loc[:,'hour'] = [int(x[0:2]) for x in act.loc[:,'time'].values]
        act = act[act['dominant_zone'].isin(np.unique(idL['id']))]
        act.to_csv(baseDir + "log/demo/" + d,index=False)

    idLs = ','.join([str(x) for x in np.unique(idL['id'])])
    ddf, fL = plib.parsePath(projDir + "single_dom_home/",id_list=idLs,is_lit=True)
    ddf = ddf.where(ddf.home_zone != -1)
    ddf = ddf.where(ddf.dominant_zone != -1)
    ddf = ddf.where(ddf['count'] > 0.0)
    ddf.show()
    pdf = ddf#.groupBy('dominant_zone','overnight_zip').sum('count')
    zipD = pdf.toPandas()
    zipD.to_csv(baseDir + "log/demo/dom_home_single.csv",index=False)
    pd.DataFrame(fL).to_csv(baseDir + "log/demo/dom_home_single_file.csv",index=False)

    idLs = ','.join([str(x) for x in np.unique(idL['id'])])
    ddf, fL = plib.parsePath(projDir + "single_zip/",id_list=idLs)
    ddf = ddf.where(ddf.overnight_zip != -1)
    ddf = ddf.where(ddf['count'] > 0.0)
    ddf.show()
    pdf = ddf.groupBy('dominant_zone','overnight_zip').sum('count')
    zipD = pdf.toPandas()
    zipD.columns = ['dominant_zone','overnight_zip','count']
    zipD.loc[:,'count'] = zipD['count']/26.
    zipD.to_csv(baseDir + "log/demo/zip_single.csv",index=False)

    idLs = ','.join([str(x) for x in np.unique(idL['id'])])
    ddf, fL = plib.parsePath(projDir + "zip_h/",id_list=idLs)
    ddf = ddf.where(ddf.overnight_zip != -1)
    ddf = ddf.where(ddf['count'] > 0.0)
    ddf.show()
    pdf = ddf.groupBy('overnight_zip').sum('count')
    zipD = pdf.toPandas()
    zipD.columns = ['overnight_zip','count']
    zipD.loc[:,'count'] = zipD['count']/26.
    zipD.to_csv(baseDir + "log/demo/zip_h.csv",index=False)

    idLs = ','.join([str(x) for x in np.unique(idL['id'])])
    ddf, fL = plib.parsePath(projDir+"zip",id_list=idLs,is_lit=True)
    ddf = ddf.withColumn("t",from_utc_timestamp(ddf.time,"CET"))
    ddf = ddf.withColumn("hour",date_format('t',"HH"))
    ddf = ddf.withColumn("wday",date_format('t',"u"))
    ddf = ddf.where(ddf.overnight_zip != -1)
    ddf = ddf.where(ddf['count'] > 0.0)
    ddf.show()
#    ddf = ddf.select(['dominant_zone','overnight_zip','hour','wday','count'])
    pdf = ddf#.groupBy('dominant_zone','overnight_zip','hour','wday').sum('count')
    pdf.show()
    zipD = pdf.toPandas()
    zipD.to_csv(baseDir + "log/demo/zip_hour.csv",index=False)
    pd.DataFrame(fL).to_csv(baseDir + "log/demo/fileList.csv")

    projDir = baseDir+"log/samba/Customer_Projects_DE/Destatis/log_no_filter/single_dom_home_h0.5/"
    dateL = os.listdir(projDir)
    for d in dateL:
        act = pd.read_csv(projDir + d,compression="gzip")
        act.columns = ['dominant_zone','home_zone','time','count']
        act = act[act['dominant_zone'].isin(np.unique(idL['id']))]
        act.to_csv(baseDir + "log/demo/" + d,index=False)


    # csv = sc.textFile(baseDir + projDir + "part-00000" )
    # rows = csv.map(lambda line: line.split(",").map(_.trim))
    # header = rows.first
    # data = rows.filter(_(0) != header(0))
    # rdd = data.map(lambda row: Row(row(0),row(1).toInt))
    # df = sqlContext.createDataFrame(rdd)

