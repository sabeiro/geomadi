import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
import pandas as pd
import numpy as np
import datetime
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
sqlContext = SQLContext(sc)

import pymongo
gradMeter = 111122.19769899677
key_file = baseDir + '/credenza/geomadi.json'
cred = []
with open(key_file) as f:
    cred = json.load(f)

if False:
    os.environ["SPARK_HOME"] = "/usr/hdp/current/spark-client"
    conf = SparkConf()
    conf.setMaster('yarn-client')
    conf.setAppName('a')
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.executor.cores", "2")
    conf.set("spark.executor.instances", "100")
    
if False: #-------------------------------demo-trips-------------------------
    projDir = "log/trips/"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = "/whitelisted_event_trips"
    outF = "raw/testrun.csv"
    
if False:#------------------------------odm-count-------------------------------
    projDir = "log/telia/single_sum"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = ""
    outF = "raw/odm_count.csv"
    df = sqlContext.read.parquet(baseDir + projDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + outF)

    tripD = tripD.sort_values(['count'],ascending=False)
    tripT = tripD.head(20)
    client = pymongo.MongoClient(cred['mongo']['address'],cred['mongo']['port'])
    coll = client["telia_se_grid"]["grid_250"]
    neiN = coll.find({'tile_id':{"$in":[int(x) for x in tripT['destination']]}})
    odmL = []
    for i in range(tripT.shape[0]):
        nei = neiN[i]
        del nei['_id']
        odmL.append(nei['geom']['coordinates'])

        with open(baseDir + "gis/telia/odm_count.geojson","w") as fo:
            fo.write(json.dumps({"type":"Feature","properties": {"name": "Most important origin","amenity": "sweeden","popupContent": "odm"},"geometry":{"type":"MultiPolygon","coordinates":odmL}},separators=(',',':')))
    for i in dateL:
        df = sqlContext.read.parquet(baseDir + projDir + i + suffixDir)
        df = df.withColumn("dir",func.lit(i))
        if i==dateL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)

    tripD = ddf.toPandas()
    tripD.to_csv(baseDir + outF)
if False:#------------------tourism----------------------
    ddf = plib.parsePath(baseDir + "log/touristen/touristen_h_dom_mcc")
    ddf = ddf.withColumn("t",from_utc_timestamp(ddf.time,"CET"))
    ddf = ddf.withColumn("hour",date_format('t',"HH"))
    ddf = ddf.withColumn("wday",date_format('t',"u"))
    pdf = ddf.groupBy('home_zone','mcc','wday','hour').sum('count')
    pdf.show()
    zipD = pdf.toPandas()
    zipD = zipD[zipD['sum(count)'] > 0.]
    zipD.to_csv(baseDir + "log/touristen/home_zone_h_national.csv",index=False)

    ddf = plib.parsePath(baseDir + "log/touristen/touristen_dom_mcc")
    ddf = ddf.withColumn("t",from_utc_timestamp(ddf.time,"CET"))
    ddf = ddf.withColumn("day",date_format('t',"d"))
    pdf = ddf.groupBy('day','home_zone','mcc').sum('count')
    pdf.show()
    zipD = pdf.toPandas()
    zipD = zipD[zipD['sum(count)'] > 0.]
    zipD.to_csv(baseDir + "log/touristen/home_zone_national.csv",index=False)

    ddf = plib.parsePath(baseDir + "log/touristen/touristen_w_dom_mcc")
    ddf = ddf.withColumn("t",from_utc_timestamp(ddf.time,"CET"))
    ddf = ddf.withColumn("day",date_format('t',"d"))
    pdf = ddf.groupBy('home_zone','mcc').sum('count')
    pdf.show()
    zipD = pdf.toPandas()
    zipD = zipD[zipD['sum(count)'] > 0.]
    zipD.to_csv(baseDir + "log/touristen/home_zone_w_national.csv",index=False)
    
    dL = os.listdir(baseDir + "log/subway/")
    for d in dL :
        df,fL = plib.parseParquet(baseDir + "log/subway/" + d)
        
        if not len(fL):
            continue
        zipD = df.toPandas()
        zipD.to_csv(baseDir + "log/subway/"+d+".csv",index=False)
    if False:
        zipP = zipD.pivot_table(index="time_destination",values="count",aggfunc=np.sum).replace(np.nan,0).reset_index()
        zipP1 = zipD.pivot_table(index="time_origin",values="count",aggfunc=np.sum).replace(np.nan,0).reset_index()
        plt.plot(zipP['time_destination'],zipP['count'])
        plt.plot(zipP1['time_origin'],zipP1['count'])
        plt.show()


    df,fL = plib.parseParquet(baseDir + "log/retail/odm_muc_3d")
    zipD = df.toPandas()
    zipD.to_csv(baseDir + "log/retail/odm_muc_3d.csv",index=False)

    df,fL = plib.parseParquet(baseDir + "log/retail/odm_muc_3o")
    zipD = df.toPandas()
    zipD.to_csv(baseDir + "log/retail/odm_muc_3o.csv",index=False)


if False:
    ddf = plib.parsePath(baseDir + "log/tampere/")
    print('loaded')
    
if False:#-----------------------demo-data--------------------------
    idL = pd.read_csv(baseDir + "raw/sel_act.csv")
    idL = idL.loc[idL['customer']=="activ"]
    projDir = baseDir+"log/samba/Customer_Projects_DE/Destatis/log_no_filter/"
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

if False:##---------------------------------jll--------------------------------
    dateL = os.listdir(baseDir + "log/jll/jll2h")
    fileN = []
    for i in dateL:
        df = sqlContext.read.parquet(baseDir + "log/jll/jll2h/" + i)
        df2 = df.groupBy('act_cell','hour').agg({'count':'sum'}).withColumnRenamed("sum(count)","count")#.collect()
        df3 = df2.groupBy("act_cell").pivot("hour").sum("count")
        dTime = re.sub('[0-9a-zA-Z_]*min_','',i)
        dTime = 'h' + dTime.replace('max_','')
        df3 = df3.withColumn("dh",func.lit(dTime))
        dfc = df2.agg({'count':'sum'}).withColumnRenamed("sum(count)","count")
        fileN.append({"dt":dTime,"n":df3.count(),"N":int(dfc.toPandas()['count'])})
        #df2.show()
        if i==dateL[0] :
            ddf = df3
        else :
            ddf = ddf.unionAll(df3)
            
    tripD = ddf.toPandas()
    tripD.to_csv(baseDir + "raw/activity.csv")
    fileN = pd.DataFrame(fileN)
    fileN.loc[:,'stay'] = fileN['dt'].apply(lambda x: x.replace('h','')).apply(lambda x: x.replace('_','-'))
    fileN.loc[:,'stay'] = fileN['stay'].apply(lambda x: -eval(x))
    fileN.loc[:,'perc'] = fileN['N']/5.980080e+06
    fileN = fileN.sort_values(['stay'])

if False:
    df.select("year", "model").save("newcars.csv", "com.databricks.spark.csv")
    win_spec=window.Window().partitionBy(['act_cell','hour'])
    df2 = ddf.withColumn('csum',func.sum(ddf.count).over(win_spec))
    ddf.show()
    ddf.registerTempTable("act_table")
    df2 = sqlContext.sql("""SELECT act_cell, hour, count, sum(count) OVER (ORDER BY act_cell, hour) as cumsum FROM act_table""")
    df.coalesce(10).write.format("com.databricks.spark.csv").save("sample.csv")

