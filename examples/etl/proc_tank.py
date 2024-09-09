import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
from dateutil import tz
import geomadi.proc_lib as plib
# import findspark
# findspark.init()
import pyspark
#sc = pyspark.SparkContext('local[*]')
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
import tarfile
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")
from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Europe/Berlin')

os.environ["SPARK_HOME"] = "/usr/hdp/current/spark-client"
conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('ttrajectory_counts')
conf.set("spark.executor.memory", "4g")
conf.set("spark.executor.cores", "4")
conf.set("spark.executor.instances", "4")
#os.environ['SPARK_HOME'] = '/usr/local/lib/python3.6/dist-packages/pyspark/'


cred = json.load(open(baseDir + "credenza/geomadi.json"))

def plog(text):
    print(text)

import importlib
importlib.reload(plib)
import shutil
#if False:
#    act = plib.parseTar(baseDir + "log/tank/","act_summer.gz")
if False:
    plog('-------------------------act-cilac-remote------------------------')
    import importlib
    importlib.reload(plib)
    mapL = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
    idL = np.unique(mapL['cilac'])
    patterN = "tank_cilac"
    dL = plib.getRemotePattern(cred['hdfs']['address'],idlist=idL,patterN="tank_cilac",isRemote=False)
    for i,d in enumerate(dL):
        print(d)
        df, hL, job = plib.parseRemoteCsv(d,idL,is_lit=False,patterN="part-00000",isRemote=False)
        newColumns = ["cilac","chi","time","count"]
        for i,c in enumerate(df.columns):
            df = df.withColumnRenamed(c,newColumns[i])
        df = df.where(df['chi'] >= 0)
        df = df.withColumn("chi",df["chi"].cast(StringType()))
        udf1 = udf(lambda x:x[:-2],StringType())
        df = df.withColumn('chi',udf1('chi'))
        fName = "_".join(d.split("_")[2:]).split(".")[0]
        tripD = df.toPandas()
        sact = tripD.pivot_table(index=["cilac","chi"],columns="time",values="count",aggfunc=np.sum).fillna(0).reset_index()
        sact.loc[:,"chi"] = sact['chi'].astype(int)
        #sact.replace(float('NaN'),0,inplace=True)
        #sact.loc[:,"id_clust"] = sact[['id_poi','chi']].apply(lambda x: "%d-%d" %(x[0],x[1]), axis=1)
        sact.sort_values(["cilac"],inplace=True)
        sact.to_csv(baseDir + "log/"+custD+"/"+fName+".csv.gz",compression="gzip",index=False)
    
if False:
    print('------------------------act-cilac-------------------------------')
    import importlib
    importlib.reload(plib)
    custD = "tank"
    mapL = pd.read_csv(baseDir + "raw/"+custD+"/map_cilac.csv")
    idlist = list(set(mapL['cilac']))
    dfL = []
    for f in ["20190122_1330_tank_cilac_t3.tar.gz","20181016_0151_tank_cilac_t2.tar.gz","20181016_0718_tank_cilac_t1.tar.gz"]:
        tar = tarfile.open(baseDir+"log/"+custD+"/act_cilac/"+f,"r:gz")
        tar.extractall(path="/tmp/")
        tar.close()
        zipD, fL = plib.parsePath("/tmp/"+f.split(".")[0],idlist=idlist,patterN="part-00000")
        zipD = zipD.withColumn('time',zipD.time.substr(0,19))
        zipD = zipD.where(col("time").isNotNull())
        zipD = zipD.where(col("count").isNotNull())
        newColumns = ["cilac","chi","time","count"]
        for i,c in enumerate(zipD.columns):
            zipD = zipD.withColumnRenamed(c,newColumns[i])
        zipD = zipD.where(zipD['chi'] >= 0)
        zipD = zipD.withColumn("chi",zipD["chi"].cast(StringType()))
        udf1 = udf(lambda x:x[:-2],StringType())
        zipD = zipD.withColumn('chi',udf1('chi'))
        fName = "_".join(f.split("_")[2:]).split(".")[0]
        tripD = zipD.toPandas()
        dfL.append(tripD)
        shutil.rmtree("/tmp/"+f)

    actD = pd.concat(dfL)
    sact = actD.pivot_table(index=["cilac","chi"],columns="time",values="count",aggfunc=np.sum).fillna(0).reset_index()
    sact.loc[:,"chi"] = sact['chi'].astype(int)
    #sact.replace(float('NaN'),0,inplace=True)
    #sact.loc[:,"id_clust"] = sact[['id_poi','chi']].apply(lambda x: "%d-%d" %(x[0],x[1]), axis=1)
    sact.sort_values(["cilac"],inplace=True)
    sact.to_csv(baseDir + "raw/tank/act_cilac.csv.gz",compression="gzip",index=False)

if False: ## act test days
    print('-------------------------act-test-days-------------------------')
    tmp1 = pd.read_csv(baseDir + "log/tank/act_test/act_11.csv.tar.gz",compression="gzip")
    tmp2 = pd.read_csv(baseDir + "log/tank/act_test/act_12.csv.tar.gz",compression="gzip")
    tmp3 = pd.read_csv(baseDir + "log/tank/act_test/act_13.csv.tar.gz",compression="gzip")
    act = pd.concat([tmp1,tmp2,tmp3],axis=0)
    tmp1 = pd.read_csv(baseDir + "log/tank/act_test/cilac_11.csv.tar.gz",compression="gzip")
    tmp1.columns = ["cilac","chi","time","act"]
    tmp2 = pd.read_csv(baseDir + "log/tank/act_test/cilac_13.csv.tar.gz",compression="gzip")
    tmp2.columns = ["cilac","chi","time","act"]
    act = pd.concat([tmp1,tmp2],axis=0)

    act = pd.read_csv(baseDir + "/log/tank/act_cilac_filtered.cvs.gz",compression="gzip")
    act = act[act['time'] == act['time']]
    act.columns = ["cilac","chi","time","act"]
    if False:
        act.loc[:,"time"] = act['time'].apply(lambda x: str(x))
        act.loc[:,"time"] = act['time'].apply(lambda x: x[:19])
        utc = act['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S'))
        act.loc[:,"time"] = utc.apply(lambda x:x.replace(tzinfo=from_zone).astimezone(to_zone).strftime("%Y-%m-%dT%H:%M:%S"))
    act = act[act['cilac'] != "-1" ]
    act = act[act['chi'] >= 0]
    sact = act.pivot_table(index=["cilac","chi"],columns="time",values="act",aggfunc=np.sum).fillna(0)
    sact = sact.reset_index()
    sact.loc[:,"chi"] = sact['chi'].astype(int)
    cellL = pd.read_csv(baseDir + "raw/tank/cilac_sel.csv")
    sact = pd.merge(sact,cellL[['cilac','X','Y','id_zone',"tech"]],on="cilac",how="left")
    #sact.replace(float('NaN'),0,inplace=True)
    sact.loc[:,"id_clust"] = sact[['id_zone','chi']].apply(lambda x: "%d-%d" %(x[0],x[1]), axis=1)
    sact.sort_values(["id_clust","cilac"],inplace=True)
    sact =  sact.drop(columns=["id_zone","chi"])
    sact.to_csv(baseDir + "raw/tank/act_cilac.csv.gz",compression="gzip",index=False)

if False: #------------------------------tank---------------------------------
    print('------------------------------activity-report-no-chirarity-------------------------')
    import importlib
    importlib.reload(plib)
    projDir = "log/tank/mc_cilac"
    projFile = "raw/tank/mc_cilac.csv.gz"
    #df = sqlContext.read.parquet(cred['hdfs']['address'] + projDir)# + repDir)
    df, fL = plib.parsePath(baseDir + projDir,is_lit=False)
    df = df.withColumn('time',df.time.substr(0,19))
    df = df.where(col("time").isNotNull())
    df = df.where(col("count").isNotNull())
    newColumns = ["id_zone","time","count"]
    for i,c in enumerate(df.columns):
        df = df.withColumnRenamed(c,newColumns[i])
    df = df.filter(df.time.like('%T%'))
    df = df.withColumn('time',df.time.substr(0,19))
    udf1 = udf(lambda x:x[:-2],StringType())
    df = df.withColumn("count",df['count'].cast('float'))
    df = df.groupby(['id_zone','time']).sum('count')
    print(df.show())
    tripD = df.toPandas()
    utc = tripD['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT'))
    tripD.loc[:,"time"] = utc.apply(lambda x:x.replace(tzinfo=from_zone).astimezone(to_zone).strftime("%Y-%m-%dT"))
    tripD.columns = ['id_zone','time','count']
    tripD.to_csv(baseDir + projFile,index=False,compression="gzip")
    print(np.unique(tripD['time'].apply(lambda x:x[:10])))


if False:
    projDir = "log/tank/act_cilac.tar.gz"
    projFile = "raw/tank/act_cilac.gz"
    cellL = pd.read_csv(baseDir + "raw/tank/cilac_sel.csv")
    df, fL = plib.parseTar(baseDir + "log/tank/","act_cilac.tar.gz",idlist=cellL['cilac'])
    df = df.withColumn('time',df.time.substr(0,19))
    df = df.where(col("time").isNotNull())
    df = df.where(col("count").isNotNull())
    newColumns = ["cilac","chi","time","count"]
    for i,c in enumerate(df.columns):
        df = df.withColumnRenamed(c,newColumns[i])
    df = df.filter(df.time.like('%T%'))
    df = df.where(df['chi'] >= 0)
    df = df.withColumn("chi",df["chi"].cast(StringType()))
    udf1 = udf(lambda x:x[:-2],StringType())
    df = df.withColumn('chi',udf1('chi'))
    df = df.withColumn('id_clust',func.concat(func.col('id_zone'),func.lit('-'),func.col('chi')))
    df = df.drop('id_zone')
    df = df.drop('chi')
    df = df.withColumn("count",df['count'].cast('float'))
    df = df.groupby(['id_clust','time']).sum('count')
    print(df.show())
    tripD = df.toPandas()
    if True:
        utc = tripD['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S'))
        tripD.loc[:,"time"] = utc.apply(lambda x:x.replace(tzinfo=from_zone).astimezone(to_zone).strftime("%Y-%m-%dT%H:%M:%S"))
    tripD.columns = ['id_clust','time','count']
    tripD.to_csv(baseDir + projFile,index=False,compression="gzip")
    print(np.unique(tripD['time'].apply(lambda x:x[:10])))
    
if True: #------------------------------tank---------------------------------
    print('------------------------------activity-report+chirarity-----------------------------')
    import importlib
    importlib.reload(plib)
    projDir = "/tdg/result/20180816_0800_tank_chirality_y30_re/"
    repDir = "*/*/*/activity_report/output/raw/"
    #df = sqlContext.read.parquet(cred['hdfs']['address'] + projDir)# + repDir)
    projDir = "log/tank/act_cilac"
    projDir = "log/tank/act_summer"
    projFile = "raw/tank/act_summer.csv.gz"
    # projDir = "log/tank/act_year_re"
    # projFile = "raw/tank/act_year_re.csv.gz"
    df, fL = plib.parsePath(baseDir + projDir,is_lit=False)
    df = df.withColumn('time',df.time.substr(0,19))
    df = df.where(col("time").isNotNull())
    df = df.where(col("count").isNotNull())
    newColumns = ["id_zone","chi","time","count"]
    for i,c in enumerate(df.columns):
        df = df.withColumnRenamed(c,newColumns[i])
    if False:
        cellL = pd.read_csv(baseDir + "raw/tank/cilac_sel.csv")
        df = df[df['id_zone'].isin(list(cellL['cilac'].values))]
    else:
        df = df.filter(df.time.like('%T%'))
        df = df.withColumn('time',df.time.substr(0,19))
        df = df.where(df['id_zone'] >= 0)
    df = df.where(df['chi'] >= 0)
    df = df.withColumn("chi",df["chi"].cast(StringType()))
    udf1 = udf(lambda x:x[:-2],StringType())
    df = df.withColumn('chi',udf1('chi'))
    df = df.withColumn('id_clust',func.concat(func.col('id_zone'),func.lit('-'),func.col('chi')))
    df = df.drop('id_zone')
    df = df.drop('chi')
    df = df.withColumn("count",df['count'].cast('float'))
    df = df.groupby(['id_clust','time']).sum('count')
    print(df.show())
    tripD = df.toPandas()
    if True:
        utc = tripD['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S'))
        tripD.loc[:,"time"] = utc.apply(lambda x:x.replace(tzinfo=from_zone).astimezone(to_zone).strftime("%Y-%m-%dT%H:%M:%S"))
    tripD.columns = ['id_clust','time','count']
    tripD.to_csv(baseDir + projFile,index=False,compression="gzip")
    print(np.unique(tripD['time'].apply(lambda x:x[:10])))

    #df.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').option("codec","org.apache.hadoop.io.compress.GzipCodec").save(baseDir + "raw/tank/act_year.csv.gz")

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

if False:
    print('---------------------chirality-age----------------------')
    df, fL = plib.parsePath(baseDir + "log/tank/chirality_age",is_lit=True)
    df = df.withColumn("t",from_utc_timestamp(df.time,"CET"))
    df = df.withColumn("wday",date_format('t',"u"))
    df.groupBy("dominant_zone","wday")
    ddf = df.groupBy("wday").sum("count")
    print(ddf.show())
    ddf = df[df.dominant_zone > 0].groupBy("wday","age").sum("count")
    ddf = df.groupBy("wday","age").sum("count")
    print(ddf.show())
    zipD = ddf.toPandas()
    zipD.loc[zipD['age']>0,"age"] = 1
    zipD = zipD.groupby(['wday','age']).agg(sum).reset_index()
    zipT = zipD.groupby(['wday']).agg(sum).reset_index()
    zipD.loc[:,'tot'] = pd.merge(zipD,zipT,on="wday",how="left")['sum(count)_y']
    zipD.loc[:,'share'] = zipD['sum(count)']/zipD['tot']
    del zipD['sum(count)']
    zipD = zipD.groupby("wday").first()
    zipD.loc[:,"tot"] = max(zipD['tot'])/zipD['tot']
    del zipD['age']
    zipD.to_csv(baseDir + "raw/tank/wday_dist.csv")

    df, fL = plib.parsePath(baseDir + "log/tank/chi_20")
    tripD = df.toPandas()
    tripD.loc[:,"time"] = tripD['time'].apply(lambda x: x[0:19])
    tripD.columns = ["id_zone","chirality","time","count"]
    tripD.loc[:,"id_clust"] = tripD[['id_zone','chirality']].apply(lambda x: str(x[0]) + "-" + x[1],axis=1)
    tripD.loc[:,"id_clust"] = tripD['id_clust'].apply(lambda x: x[0:-2])
    tripD = tripD.pivot_table(index="id_clust",columns="time",values="count",aggfunc=np.sum).replace(np.nan,0).reset_index()
    tripD.to_csv(baseDir + "raw/tank/act_march_20.csv.gz",index=False,compression="gzip")

if False:
    print('----------------------------dir-count-----------------------------')
    projDir = baseDir + "log/tank/dir_count/"
    fL = []
    for path,dirs,files in os.walk(projDir):
        for f in files:
            if re.search("dir_count",f):
                fL.append(path+f)
    for i in fL:
        dirD = pd.read_csv(i,sep="|")
        dirD.dropna(inplace=True)
        if i == fL[0]:
            dirS = dirD
        else:
            dirS = pd.concat([dirS,dirD],axis=0)
    cL = dirS.columns
    cL = [re.sub(" ","",x) for x in cL]
    dirS.columns = cL
    dirS.loc[:,"time"] = dirS['ts_start'].apply(lambda x: x[1:11]+"T00:00:00")
    for i in [x for x in dirS.columns if bool(re.search("speed",x))] + ['ts_start']:
        del dirS[i]
    dirG = dirS.groupby(["tile_id","time"]).agg(sum).reset_index()
    dirG.to_csv(baseDir + "raw/tank/dir_count.csv",index=False)
    
    projDir = "log/tank/"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = ""
    outF = "raw/dir_count.csv"
    df = sqlContext.read.parquet(baseDir + projDir + "dir_count" + suffixDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + outF)

    projDir = "log/tank/"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = ""
    df = sqlContext.read.parquet(baseDir + projDir + "angle" + suffixDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "raw/tank/act_angle_13.csv.gz",compression="gzip",index=False)

    idL = pd.read_csv(baseDir + "raw/cilac_sel.csv")
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idL['cilac'])])
    importlib.reload(plib)
    df, fL = plib.parsePath(baseDir + "log/tank/prod_code",id_list=idLs,is_lit=True)
    df = df.where(df.dominant_zone != "-1")
    df.show()
    for i,f in enumerate(fL):
        pdf = df.where(df.dir == i)
        actD = pdf.toPandas()
        actD.to_csv(baseDir + "log/tank/filter/" + f + ".csv",index=False)
        print(f)

if False: #------------------------------tank---------------------------------
    ##------------------------------activity-report+chirarity-----------------------------
    import importlib
    importlib.reload(plib)

    df, fL = plib.parsePath(baseDir + "log/tank/act_year",is_lit=True)
    tripD = df.toPandas()
    del tripD['dir']
    tripD.columns = ["id_zone","chi","time","count"]
    tripD = tripD[tripD['time'] == tripD['time']] ## no time aggregation
    tripD.loc[:,"time"] = tripD['time'].apply(lambda x: x[:19])
    tripD = tripD[[bool(re.search("T",x)) for x in tripD['time']]]
    tripD.loc[:,"chi"] = tripD['chi'].apply(lambda x: x[:-2])
    tripD.loc[:,"id_zone"] = tripD['id_zone'].astype(int)
    tripD = tripD[tripD['id_zone'] >= 0]
    tripD.loc[:,"id_clust"] = tripD[['id_zone','chi']].apply(lambda x: "%d-%s" % (x[0],x[1]),axis=1)
    tripD.loc[:,"day"]  = tripD['time'].apply(lambda x:x[0:10])
    print(np.unique(tripD['day']))
    print(len(np.unique(tripD['day']))/7.)
    del tripD['id_zone'], tripD['chi'], tripD['day']
    tripD = tripD.groupby(["id_clust","time"]).agg(sum).fillna(0).reset_index()
#    tripD.loc[:,"id_zone"] = act['id_zone'].apply(lambda x: x.split("-")[0])
    tripD.to_csv(baseDir + "raw/tank/act_year.csv.tar.gz",index=False,compression="gzip")
    
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
    tripD.to_csv(baseDir + "raw/tank/act_month.csv.tar.gz",index=False,compression="gzip")

    df, fL = plib.parsePath(baseDir + "log/tank/chirality_age",is_lit=True)
    df = df.withColumn("t",from_utc_timestamp(df.time,"CET"))
    df = df.withColumn("wday",date_format('t',"u"))
    df.groupBy("dominant_zone","wday")
    ddf = df.groupBy("wday").sum("count")
    print(ddf.show())
    ddf = df[df.dominant_zone > 0].groupBy("wday","age").sum("count")
    ddf = df.groupBy("wday","age").sum("count")
    print(ddf.show())
    zipD = ddf.toPandas()
    zipD.loc[zipD['age']>0,"age"] = 1
    zipD = zipD.groupby(['wday','age']).agg(sum).reset_index()
    zipT = zipD.groupby(['wday']).agg(sum).reset_index()
    zipD.loc[:,'tot'] = pd.merge(zipD,zipT,on="wday",how="left")['sum(count)_y']
    zipD.loc[:,'share'] = zipD['sum(count)']/zipD['tot']
    del zipD['sum(count)']
    zipD = zipD.groupby("wday").first()
    zipD.loc[:,"tot"] = max(zipD['tot'])/zipD['tot']
    del zipD['age']
    zipD.to_csv(baseDir + "raw/tank/wday_dist.csv")

    df, fL = plib.parsePath(baseDir + "log/tank/chi_20")
    tripD = df.toPandas()
    tripD.loc[:,"time"] = tripD['time'].apply(lambda x: x[0:19])
    tripD.columns = ["id_zone","chirality","time","count"]
    tripD.loc[:,"id_clust"] = tripD[['id_zone','chirality']].apply(lambda x: str(x[0]) + "-" + x[1],axis=1)
    tripD.loc[:,"id_clust"] = tripD['id_clust'].apply(lambda x: x[0:-2])
    tripD = tripD.pivot_table(index="id_clust",columns="time",values="count",aggfunc=np.sum).replace(np.nan,0).reset_index()
    tripD.to_csv(baseDir + "raw/tank/act_march_20.csv.tar.gz",index=False,compression="gzip")

    
    ##----------------------------dir-count-----------------------------
    projDir = baseDir + "log/tank/dir_count/"
    fL = []
    for path,dirs,files in os.walk(projDir):
        for f in files:
            if re.search("dir_count",f):
                fL.append(path+f)
    for i in fL:
        dirD = pd.read_csv(i,sep="|")
        dirD.dropna(inplace=True)
        if i == fL[0]:
            dirS = dirD
        else:
            dirS = pd.concat([dirS,dirD],axis=0)
    cL = dirS.columns
    cL = [re.sub(" ","",x) for x in cL]
    dirS.columns = cL
    dirS.loc[:,"time"] = dirS['ts_start'].apply(lambda x: x[1:11]+"T00:00:00")
    for i in [x for x in dirS.columns if bool(re.search("speed",x))] + ['ts_start']:
        del dirS[i]
    dirG = dirS.groupby(["tile_id","time"]).agg(sum).reset_index()
    dirG.to_csv(baseDir + "raw/tank/dir_count.csv",index=False)
    
    projDir = "log/tank/"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = ""
    outF = "raw/dir_count.csv"
    df = sqlContext.read.parquet(baseDir + projDir + "dir_count" + suffixDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + outF)

    projDir = "log/tank/"
    dateL = os.listdir(baseDir + projDir)
    suffixDir = ""
    df = sqlContext.read.parquet(baseDir + projDir + "angle" + suffixDir)
    tripD = df.toPandas()
    tripD.to_csv(baseDir + "raw/tank/act_angle_13.csv.tar.gz",compression="gzip",index=False)

    idL = pd.read_csv(baseDir + "raw/cilac_sel.csv")
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idL['cilac'])])
    importlib.reload(plib)
    df, fL = plib.parsePath(baseDir + "log/tank/prod_code",id_list=idLs,is_lit=True)
    df = df.where(df.dominant_zone != "-1")
    df.show()
    for i,f in enumerate(fL):
        pdf = df.where(df.dir == i)
        actD = pdf.toPandas()
        actD.to_csv(baseDir + "log/tank/filter/" + f + ".csv",index=False)
        print(f)

