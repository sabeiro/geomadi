"""
proc_lib:
spark utils to process and search for the output
"""

import os, json, re
import pandas as pd
import numpy as np
import datetime
import subprocess
import matplotlib.pyplot as plt

os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-1.8.0-openjdk-amd64'
baseDir = os.environ['LAV_DIR']
# import findspark
# findspark.init()
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col

conf = (SparkConf()
    .setMaster("yarn-client")
    .setAppName("proc library")
    .set("spark.deploy-mode", "cluster"))
conf.set("spark.executor.memory", "10g")
conf.set("spark.executor.cores", "10")
conf.set("spark.executor.instances", "2")
conf.set("spark.driver.maxResultSize", "10g")
conf.set("spark.driver.memory", "10g")
conf.set("spark.sql.crossJoin.enabled", "true")

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

import tarfile

key_file = baseDir + '/credenza/geomadi.json'
cred = json.load(open(baseDir + "credenza/geomadi.json"))

def getHdfsPattern(projDir,is_lit=False,patterN="part-00000",isRemote=False):
    """download file list from hdfs"""
    if isRemote:
        cmd = 'ssh cr_hu "hdfs dfs -ls -R '+cred['hdfs']['address']+projDir
    else:
        cmd = 'hdfs dfs -ls -R '+projDir
    files = str(subprocess.check_output(cmd,shell=True))
    dL = [x.split(" ")[-1] for x in files.strip().split('\\n')]
    dL = [x for x in dL if not bool(re.search("running",x))]
    dL = [x for x in dL if bool(re.search(patterN,x))]
    dL = ["/".join(x.split("/")[:-2]) for x in dL] 
    dL = list(set(dL))
    return dL

def parsePath(projDir,idlist=[None],is_lit=False,patterN="part-00000",fL=[None]):
    """collects all folder outputs into a single data frame"""
    idlist = list(idlist)
    id_list = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
    if not any(fL):
        fL = []
        for path,dirs,files in os.walk(projDir):
            for f in files:
                if re.search(patterN,f):
                    fL.append(path+"/"+f)
    fL = [x for x in fL if not bool(re.search("^\.",x.split("/")[-1]))]
    for i,f in enumerate(fL):
        print(f)
        print("%f" % (float(i)/float(len(fL))),end='\r',flush=True)
        try:
            df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load("file://"+f)
        except:
            print('not read')
            continue
        if any(idlist):
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN (" + id_list + ")")
        if is_lit:
            df = df.withColumn("dir",func.lit(i))
        if f==fL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)
    print('loaded')
    fL1 = fL
    fL1 = [re.sub(projDir,"",x) for x in fL1]
    fL1 = [re.sub("/part-00000","",x) for x in fL1]
    fL1 = [re.sub("/","-",x) for x in fL1]
    fL1 = [re.sub("^-","",x) for x in fL1]
    return ddf, fL1

def parseHdfsCsv(projDir,idlist=[None],is_lit=False,patterN="part-00000",isRemote=False,isAgg=True):
    idlist = list(idlist)
    id_list = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
    nTree = 12 if isAgg else 11
    if isRemote:
        cmd = 'ssh cr_hu "hdfs dfs -ls -R '+cred['hdfs']['address']+cred['hdfs']['output']+projDir+'"'
    else:
        cmd = 'hdfs dfs -ls -R '+cred['hdfs']['output']+projDir
    files = str(subprocess.check_output(cmd,shell=True))
    fL = [x.split(" ")[-1] for x in files.strip().split('\\n')]
    jobL = [x for x in fL if bool(re.search("json",x))]
    if isRemote:
        cmd = 'ssh cr_hu "hdfs dfs -cat '+jobL[0]+'"'
    else :
        cmd = 'hdfs dfs -cat '+jobL[0]
    cat = str(subprocess.check_output(cmd,shell=True))
    for i in ["\\\\n","\'","\\\\t","\\\\"]:
        cat = re.sub(i,"",cat)
    job = json.loads(cat[1:])
    nDay = 1./float(len(job['date_list']))
    fL = [x for x in fL if bool(re.search("part-00000",x))]
    fL = [x for x in fL if len(x.split("/")) == nTree]
    hL = [x.split("/") for x in fL]
    #hL = ["%s-%s-%s" % (x[6],x[7],x[8]) for x in hL]
    for f in fL:
        df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(f)
        df = df.withColumn('time',df.time.substr(0,19))
        #df = df.where(col("time").isNotNull())
        df = df.where(col("count").isNotNull())
        df = df.where(df['count'] > 0)
        if isAgg:
            df = df.withColumn('count',df['count']*nDay)
        if any(idlist):
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN ("+id_list+")")
        if f==fL[0] : ddf = df
        else : ddf = ddf.unionAll(df)
    return ddf, hL, job

def parseParquet(projDir,idlist=[None],is_lit=False,patterN="part-00000",dL=[None]):
    """"parse parquet files from file list"""
    if not any(dL):
        dL = []
        for path,dirs,files in os.walk(projDir):
            for f in files:
                if re.search(patterN,f):
                    dL.append(path+"/")
                    break
    if not len(dL):
        return [],[]
    
    for i,d in enumerate(dL):
        print("%.2f%%" % (float(i)/float(len(dL))*100.),end="\r",flush=True)
        # fL = os.listdir(d)
        # fL = [x for x in fL if bool(re.search(patterN,x))]
        # fL = [x for x in fL if not bool(re.search("^\.",x.split("/")[-1]))]
        # for f in fL:
        df = sqlContext.read.parquet(d)
        if is_lit: df = df.withColumn("dir",func.lit(i))
        if i == 0: ddf = df
        else : ddf = ddf.unionAll(df)
    print("loaded {}".format(projDir))
    fL = [re.sub(projDir,"",x) for x in dL]
    return ddf, fL

def parseTarParquet(dname,fname,idlist=[None],patterN="part-00000",is_lit=False):
    idlist = list(idlist)
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
    print(fname)
    try : 
        tar = tarfile.open(dname+fname, "r:gz")
    except:
        return 0
    csvL = [x for x in tar.getnames()]# if bool(re.search(patterN,x))]
    jobL = [x for x in csvL if bool(re.search("json",x))]
    for i,cs in enumerate(csvL):
        print("%f" % (float(i)/float(len(csvL))),end='\r',flush=True)
        t = tar.getmember(cs)
        tar.extract(path="/tmp/"+cs)
    job = json.load(open(jobL[0]))
    csvL = np.unique(["/".join(x.split("/")[:-1]) for x in csvL])
    maxL = max([len(x.split("/")) for x in csvL])
    csvL = [x for x in csvL if len(x.split("/")) == maxL]
    for i,cs in enumerate(csvL):
        print("%f" % (float(i)/float(len(csvL))),end='\r',flush=True)
        df = sqlContext.read.parquet(cs)
        if isLit:
            df = df.withColumn("dir",func.lit(i))
        if any(idlist):
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE origin IN ("+idLs+") OR destination IN (" + idLs + ")")
            #df = df[df['dominant_zone'].isin(idlist:_*))]
        if cs==csvL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)
    csvL = [x for x in tar.getnames() if bool(re.search(patterN,x))]
    csvL = np.unique(["/".join(x.split("/")[:-1]) for x in csvL])
    csvL = [x for x in csvL if len(x.split("/")) == maxL]
    tar.close()
    return ddf, csvL, job

def parseTar(dname,fname,idlist=[None],patterN="part-00000",isLit=False):
    idlist = list(idlist)
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
    print(fname)
    try : 
        tar = tarfile.open(dname+fname, "r:gz")
    except:
        return 0
    csvL = [x for x in tar.getnames() if bool(re.search(patterN,x))]
    cs = csvL[0]
    for i,cs in enumerate(csvL):
        print("%f" % (float(i)/float(len(csvL))),end='\r',flush=True)
        t = tar.getmember(cs)
        tar.extract(cs)
        df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(cs)
        if any(idlist):
            sqlContext.registerDataFrameAsTable(df,"table1")
            df = sqlContext.sql("SELECT * FROM table1 WHERE dominant_zone IN (" + idLs + ")")
            #df = df[df['dominant_zone'].isin(idlist:_*))]
        if isLit:
            df = df.withColumn("dir",func.lit(i))
        if cs==csvL[0] :
            ddf = df
        else :
            ddf = ddf.unionAll(df)
    tar.close()
    return ddf, csvL

def parseTarPandas(dname,fname,idlist=[None],patterN="part-00000"):
    idlist = list(idlist)
    idLs = ','.join(['"'+str(x)+'"' for x in np.unique(idlist)])
    print(fname)
    try : 
        tar = tarfile.open(dname+fname, "r:gz")
    except:
        return 0
    csvL = [x for x in tar.getnames() if bool(re.search(patterN,x))]
    cs = csvL[0]
    for i,cs in enumerate(csvL):
        print("%f" % (float(i)/float(len(csvL))),end='\r',flush=True)
        t = tar.getmember(cs)
        tar.extract(cs)
        df = pd.read_csv(cs)
        if any(idlist):
            df = df[df['dominant_zone'].isin(idlist)]
        if cs==csvL[0] :
            ddf = df
        else :
            ddf = pd.concat([ddf,df],axis=0)
    tar.close()
    return ddf, csvL


def writeCsv(df,fName):
    try:
        df.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save(fName)
    except:
        dp = df.toPandas()
        dp.to_csv(fName,index=False)


def readRemote(projDir):
    projDir = cred['hdfs']['address'] + projDir
    df = sqlContext.read.parquet(projDir)
    return df.toPandas()

def readRemoteCsv(projDir):
    projDir = cred['hdfs']['address'] + projDir
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load(projDir)
    return df

def browseFS(projDir):
    URI = sc._gateway.jvm.java.net.URI
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    fs = FileSystem.get(URI(cred['hdfs']['address']), sc._jsc.hadoopConfiguration())
    status = fs.listStatus(Path(projDir))
    fL = [x.getPath().getName() for x in status]
    return fL

def dateList(projDir,dL):
    jL = []
    for d in dL:
        i = os.listdir(projDir+"/"+d+"/")
        i = [x for x in i if bool(re.search("json",x))][0]
        j = json.load(open(projDir+"/"+d+"/"+i))
        l = len(j['date_list'])
        jL.append({"dir":d,"nDate":l})
    return jL
