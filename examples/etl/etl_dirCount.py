import os, sys, gzip, random, csv, json, datetime, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import postgres
import psycopg2
import geomadi.proc_lib as plib
from dateutil import tz
from datetime import timezone
import scipy.stats as stats
from_zone = tz.gettz('UTC')
to_zone = tz.gettz('Europe/Berlin')
from bs4 import BeautifulSoup
import urllib3
import requests
import tarfile
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.functions import col, lit, coalesce, greatest
from pyspark.sql.types import *
from pyspark.sql import Row
import shutil
import pymongo
sqlContext = SQLContext(sc)

os.environ["SPARK_HOME"] = "/usr/hdp/current/spark-client"
conf = SparkConf()
conf.setMaster('yarn-client')
conf.setAppName('ttrajectory_counts')
conf.set("spark.executor.memory", "16g")
conf.set("spark.executor.cores", "16")
conf.set("spark.executor.instances", "16")

print('---------------------------------direction-counts--------------------------')
custD = "mc"
custD = "bast"
custD = "tank"
idField = "id_poi"

cred = json.load(open(baseDir + "credenza/geomadi.json","r"))
poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")

if True:
    print('--------------------retrieve-date-list-----------------')
    resq = requests.get(cred['processed']['address'])
    soup = BeautifulSoup(resq.text,'lxml')# 'html.parser')
    table = soup.find('table')
    rows = table.findAll('tr')
    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)

    rows = table.findAll('th')
    rows = [row.text.strip() for row in rows]
    proc = pd.DataFrame(data,columns=rows)
    proc = proc[proc['ma'] != '']
    proc = proc[proc['Data day'] >= "2018-09-01"]
    proc = proc[proc['Data day'] >= "2018-12-31"]
    proc = proc[proc['Data day'] <=  "2019-01-31"]
    #proc = proc[proc['Data day'] >  "2019-02-31"]
    dateL = np.unique(proc['Data day'])
    if False:
        dateL = pd.DataFrame({"day":dateL})
        dateL.to_csv(baseDir + "raw/"+custD+"/dateList.csv",index=False)

if False:
    print('----------------prepare-postgres-query-------------------')
    poi = pd.read_csv(baseDir+"raw/"+custD+"/poi.csv")
    tileL = [x for x in np.unique(poi['id_tile']) if x == x]
    if False:
        tileM = pd.read_csv(baseDir + "raw/bast/id_bast_tile.csv")
        tileL = [x for x in np.unique(tileM['id_tile']) if x == x]
        custD = "bast"
    tileL = [int(x) for x in tileL]
    tileS = ", ".join([str(x) for x in tileL])
    cred = json.load(open(baseDir + "credenza/geomadi.json","r"))
    queL = ""
    dayL = ["201809%02d"%i for i in range(1,31)]
    dayL = [re.sub("-","",i) for i in dateL]
    for i in reversed(dayL):
        queI = "psql -p "+cred['postgres']['port']+" -U "+cred['postgres']['user']+" -d "+cred['postgres']['db']+" -h "+cred['postgres']['host']+' -c "'
        queI += "COPY (SELECT * FROM "+cred['postgres']['table']+i+" WHERE tile_id in ("+tileS+") ) TO '/tmp/dirCount_"+i+".csv' "+'(format csv,header true);"'
        queL += queI + "\n"

    queL += "cp /tmp/dirCount_* "+baseDir+"log/"+custD+"/dirCount/"
    open(baseDir+"log/"+custD+"/"+"dirCount_query.sh","w").write(queL)

if False:
    print('------------------return-tiles-with-counts---------------')
    i = '20190301'
    queI = "psql -p "+cred['postgres']['port']+" -U "+cred['postgres']['user']+" -d "+cred['postgres']['db']+" -h "+cred['postgres']['host']+' -c "'
    que = "SELECT dirc.tile_id, SUM(dirc.out) AS out FROM (SELECT tile_id, ts_start, COALESCE(west_out,0) + COALESCE(east_out,0) + COALESCE(north_out,0) + COALESCE(south_out,0) AS out FROM "+cred['postgres']['table']+i+") AS dirc WHERE (dirc.out > 0) GROUP BY dirc.tile_id"
    queI += "COPY ("+que+") TO '/tmp/tileDistinct_"+i+".csv' "+'(format csv,header true);"'
    print(queI)
    

if False:
    print('------------------parse-dir-count----------------------')
    import importlib
    importlib.reload(plib)
    minf = lit(float("-inf"))
    poi  = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
    projDir = baseDir+"log/"+custD+"/dirCount/"
    if custD == "tank":
        poi = poi[poi['competitor'] == 0]
    poi = poi[poi['id_tile'] == poi['id_tile']]
    poi.loc[:,"id_tile"] = poi['id_tile'].astype(int)
    poi = poi.groupby("id_poi").first().reset_index()
    poi.loc[:,"orientation_manual"] = poi['orientation_manual'].apply(lambda x: x + "_out")
    dL = os.listdir(projDir)
    dL = [x for x in dL if bool(re.search("dirCount",x))]
    dL = [x for x in dL if bool(re.search("tar.gz",x))]
    schema = StructType([StructField(idField,StringType()),StructField('tile_id',IntegerType()),StructField("orientation",StringType())])
    rdd = sc.parallelize([tuple(x) for x in poi[[idField,"id_tile","orientation_manual"]].values])
    poiT = sqlContext.createDataFrame(rdd,schema)
    def get_col(row):
        dicta = row.asDict()
        orientation = dicta["orientation"]
        desired = dicta[orientation]
        dicta["out"] = desired
        return Row(**dicta)
    dirL = []
    dL = ['dirCount_19-01.tar.gz','dirCount_19-02.tar.gz','dirCount_19-03.tar.gz']
    for d in dL:
        tar = tarfile.open(projDir+d,"r:gz")
        tar.extractall(path="/tmp/")
        tar.close()
        fL = os.listdir("/tmp/dirCount/")
        dfL = []
        for f in fL:
            df = pd.read_csv("/tmp/dirCount/"+f)
            tL = [x for x in df.columns if bool(re.search("_in$",x)) ]
            df.loc[:,"max"] = df[tL].apply(lambda x: max(x),axis=1)
            df.loc[:,"sum"] = df[tL].apply(lambda x: max(x),axis=1)
            tL = ['tile_id','ts_start','max','sum','east_out','north_out','south_out','west_out']
            df = df[tL]
            dfL.append(df)
        dirT = pd.concat(dfL)
        # dirT, fL = plib.parsePath("/tmp/dirCount",patterN="dirCount")
        # tL = ['tile_id','ts_start','max','sum','east_out','north_out','south_out','west_out']
        # dirT = dirT.select([c for c in dirT.columns if c in tL])
        # tL = [x for x in dirT.columns if bool(re.search("_out$",x)) ]
        # rowmax = greatest(*[coalesce(col(x),minf) for x in tL])
        # dirT = dirT.withColumn("max", rowmax)
        # dirT = dirT.withColumn("sum",sum(dirT[col] for col in tL))
        # dirT = dirT.join(poiT,dirT['tile_id'] == poiT['tile_id'],how="left")
        # dirT = dirT.rdd.map(get_col).toDF()
        # dirT = dirT.toPandas()
        shutil.rmtree("/tmp/dirCount")
        dirT.rename(columns={"tile_id":"id_tile","max":"dircMax","sum":"dircSum","ts_start":"time"},inplace=True)
        dirT['time'] = dirT['time'].apply(lambda x: x[:19])
        utc = dirT['time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        dirT.loc[:,"time"] = utc.apply(lambda x: x.replace(tzinfo=from_zone).astimezone(to_zone).strftime("%Y-%m-%dT%H:%M:%S"))
        dirL.append(dirT)
        
    dirT = pd.concat(dirL)
    dirT = pd.merge(dirT,poi[['id_tile','id_poi','orientation_manual']],on="id_tile",how="left")
    dirT = dirT[dirT['orientation_manual'] == dirT['orientation_manual']]
    dirT.loc[:,"out"] = dirT.apply(lambda x: x[x['orientation_manual']],axis=1)

    def aggF(x):
        norm = 0.4226623389831198
        foot = sum(x)/len(x)*norm*1.1*24.
        return foot

    dirc = dirT.pivot_table(index="id_tile",columns="time",values="out",aggfunc=np.sum)
    dirc = pd.merge(dirc,poi[['id_poi','id_tile']],on="id_tile",how="left")
    hL = dirc.columns[[bool(re.search('-??T',x)) for x in dirc.columns]]
    dirc = dirc[dirc['id_poi'] == dirc['id_poi']]
    dirc = dirc[["id_poi"] + list(hL)]
    dirc.to_csv(baseDir + "raw/"+custD+"/dirCount_h.csv.gz",compression="gzip",index=False)

    dirT.loc[:,"day"] = dirT['time'].apply(lambda x: x[:11])
    dird = dirT.pivot_table(index="id_tile",columns="day",values="out",aggfunc=np.sum).reset_index()
    dird.loc[:,"id_tile"] = dird['id_tile'].astype(int)
    dird = pd.merge(dird,poi[['id_poi','id_tile']],on="id_tile",how="left")
    hL = dird.columns[[bool(re.search('-??T',x)) for x in dird.columns]]
    dird = dird[dird['id_poi'] == dird['id_poi']]
    dird = dird[["id_poi"] + list(hL)]
    dird.to_csv(baseDir + "raw/"+custD+"/dirCount_d.csv.gz",compression="gzip",index=False)
    if False:
        hL = dirc.columns[[bool(re.search('-??T',x)) for x in dirc.columns]]
        y = dirc[hL].sum(axis=0)
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S") for x in hL]
        plt.plot(t,y)
        plt.show()

    if False:
        hL = dird.columns[[bool(re.search('-??T',x)) for x in dird.columns]]
        y = dird[hL].sum(axis=0)
        t = [datetime.datetime.strptime(x,"%Y-%m-%dT") for x in hL]
        plt.plot(t,y)
        plt.show()
        
if False:
    print('-------------------call-postgres-directly----------------------')
    try:
        connection = psycopg2.connect(dbname=cred['postgres']['db'],user=cred['postgres']['user'],password=cred['postgres']['password'],host=cred['postgres']['host'],port=cred['postgres']['port'])
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
    print("You are connected to - ", record)
    except (Exception, psycopg2.Error) as error :
        print ("Error while connecting to PostgreSQL", error)
        
    cursor.execute("select * from tile_direction_daily_hours_sum_20180710 limit 10;")
    cursor.execute("select * from tile_direction_daily_hours_sum_20180710 where tile_id in ("+tileS+");")
    print(cursor.fetchone())
    importlib.reload(gen)

if False:
    print('-------------------------bast-isocalendar--------------------------')
    bL = []
    for f in ["bast17","bast16","bast15"]:
        bast = pd.read_csv(baseDir + "log/bast/"+f+".csv.gz",compression="gzip")
        bast.loc[:,"date"] = bast['date'].astype(str)
        bast.loc[:,"time"] = bast["date"].apply(lambda x: "20%s-%s-%sT" %(x[:2],x[2:4],x[4:6]) )
        bast.loc[:,"time"] = bast[["time","hour"]].apply(lambda x: x[0]+"%02d:00:00" %(int(x[1])-1),axis=1 )
        ical = bast['time'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%dT%H:%M:%S").isocalendar() )
        bast.loc[:,"isocal"] = ["%02d-%02d" % (x[1],x[2]) for x in ical]
        bL.append(bast)
    bast = pd.concat(bL)
    bast.loc[:,"dir"] = bast['dir1'] + bast['dir2']
    past = bast.pivot_table(index="id_bast",columns="isocal",values="dir",aggfunc=np.sum)
    past.to_csv(baseDir + "raw/basics/bast_iso.csv.gz",compression="gzip")
    for i in ['tank','mc']:
        poi = pd.read_csv(baseDir + "raw/"+i+"/poi.csv")
        poi = poi.groupby("id_poi").first().reset_index()
        last = pd.merge(past,poi[['id_bast','id_poi']],how="left",on="id_bast")
        last = last[last['id_poi'] == last['id_poi']]
        last.loc[:,"id_poi"] = last['id_poi'].astype(int)
        last.drop(columns=["id_bast"],inplace=True)
        last.to_csv(baseDir + "raw/"+i+"/bast_iso.csv.gz",compression="gzip",index=False)
    
if False:
    print('--------------------------prepare-cassandra-query-----------------------')
    mist = pd.read_csv(baseDir + "raw/tank/visit_march_melted.csv.tar.gz",compression="gzip")
    tileL = pd.read_csv(baseDir + "raw/tank/tileList.csv")
    tileL = tileL[['tile_id','id_poi','id_clust']]
    #tileU = np.unique(poi['id_tile'])
    tileU = np.unique(tileL['tile_id'])
    tileU = tileU[tileU == tileU]
    tileU = [int(x) for x in tileU]
    dateU = np.unique(mist['time'])
    dateU = [x[0:4] + x[5:7] + x[8:10] for x in dateU]
    cred = json.load(open(baseDir + "credenza/geomadi.json","r"))
    query = ""
    for i in dateU:
        query += "cqlsh " + cred['cassandra']['address'] + " -e \'select * from " + cred['cassandra']['keyspace'] +"."+cred['cassandra']['table']+str(i)+" where tile_id in ("
        for j in tileU:
        query += "" + str(j) + ","
        query = query[:-1] + ");\' > dir_count_"+str(i)+".dat\n"
    open(baseDir + "raw/tank/cassandra.cql","w").write(query)

    
print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

