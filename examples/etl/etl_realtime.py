import os, sys, gzip, random, csv, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import lernia.train_viz as t_v
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import seaborn as sns
import urllib3,requests
import base64

idField = "id_poi"
custD = "realtime"

print('------------------------load-reshape------------------------')

poi = pd.read_csv(baseDir + "raw/"+custD+"/poi.csv")
cred = json.load(open(baseDir + "credenza/geomadi.json"))
baseUrl = cred['realtime']['address'] + cred['realtime']['url']
headers = {"Content-type":"application/json; charset=UTF-8","Accept":"application/json","User_Agent":"Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10.6; en-US; rv:1.9.1) Gecko/20090624 Firefox/3.5"}
auth_string = (cred['realtime']['user']+":"+cred['realtime']['pass']).encode()
headers['Authorization'] = "Basic "+base64.standard_b64encode(auth_string).decode()

start = int(datetime.datetime(2019,6,14,0,0).timestamp())
end = int(datetime.datetime.today().timestamp())

aggL = []
for i,g in poi.iterrows():
    print("complete %.2f" % (i/poi.shape[0]) )
    url = baseUrl + "from/"+str(start)+"/to/"+str(end)+"?location_id=" + g[idField]
    resq = requests.get(url,headers=headers,verify=False)
    if resq.status_code >= 300:
        print("missing poi %s" % g[idField])
        continue
    res = resq.json()['results']
    for j in res:
        crm = j['metrics']['crm']
        cross = j['metrics']['crm_cross']
        k = {'date':j['date'],'id_poi':j['location_id'],'crm':j['metrics']['crm'],'crm_cross':j['metrics']['crm_cross']}
        aggL.append(pd.DataFrame(k))
    
aggL = pd.concat(aggL).reset_index()
crmT = list(np.unique(aggL['index']))
cL = [x for x in crmT if bool(re.search("_gender",x))]
lL = [x for x in crmT if not bool(re.search("_gender",x))]
cross = aggL.pivot_table(index=['id_poi','date'],columns='index',values='crm_cross',aggfunc=np.nansum).reset_index()
cross = cross[ [idField,"date"] + cL]
crm = aggL.pivot_table(index=['id_poi','date'],columns='index',values='crm',aggfunc=np.nansum).reset_index()
crm = crm[ [idField,"date"] + lL]

crm = crm.replace(float('nan'),0)
crm.loc[:,"age_5"] = crm['age_5'] + crm['age_6']
del crm['age_6']

cross = cross.replace(float('nan'),0)
cross.loc[:,"age_5_gender_1"] = cross['age_5_gender_1'] + cross['age_6_gender_1']
cross.loc[:,"age_6_gender_2"] = cross['age_5_gender_2'] + cross['age_6_gender_2']
del cross['age_6_gender_2'], cross['age_6_gender_1']

cross.to_csv(baseDir + "raw/"+custD+"/poi_realtime_cross.csv.gz",compression="gzip",index=False)
crm.to_csv(baseDir + "raw/"+custD+"/poi_realtime_crm.csv.gz",compression="gzip",index=False)

if False:
    print('-------------------------rebin-age-class-census-data------------------------')
    census = pd.DataFrame(json.load(open(baseDir + "raw/basics/census.json","r"))['germany'])
    census.loc[:,"male"] = census['male'].apply(lambda x: float(re.sub("%","",x)))
    census.loc[:,"female"] = census['female'].apply(lambda x: float(re.sub("%","",x)))
    census.loc[:,"age"]  = "age_5"
    census.loc[census['variable'].isin(['0-4','5-9','10-14','15-19']),"age"] = "age_1"
    census.loc[census['variable'].isin(['20-24','25-29']),"age"]  = "age_1"
    census.loc[census['variable'].isin(['30-34','35-39']),"age"]  = "age_2"
    census.loc[census['variable'].isin(['40-44','45-49']),"age"]  = "age_3"
    census.loc[census['variable'].isin(['50-54','55-59']),"age"]  = "age_4"
    census.loc[census['variable'].isin(['60-64','65-69']),"age"]  = "age_5"
    census.loc[census['variable'].isin(['70-74','75-79','80-84','85-89','90-94','95-99','100+']),"age"]  = "age_5"
    census.loc[:,"share"] = (census['male'] + census['female'])*.5
    census.loc[:,"share"] = census['share']/census['share'].sum()
    print(census.groupby(['age']).agg(sum))

    celG = pd.DataFrame({"male":[.66],"female":[.63]})
    celA = pd.DataFrame({"18-29":[.85],"30-49":[.79],"50-64":[.54],"65+":[.27]})
    celA = pd.DataFrame({"18-29":[.94],"30-49":[.89],"50-64":[.73],"65+":[.46]})
    celA = pd.DataFrame({"age_1":[.94],"age_2":[.90],"age_3":[.89],"age_4":[.77],"age_5":[.73]}).T.reset_index()
    celA.columns = ['age','corr']
    census = census.merge(celA,on="age",how="left")
    census = census.replace(float('nan'),1.)
    census.loc[:,"share"] = census['share']*census['corr']
    
    cenA = census.pivot_table(columns="age",values="share",aggfunc=np.sum)
    cenA.to_csv(baseDir + "raw/"+custD+"/age_census.csv",index=False)

if False:
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    df = spark.read.parquet(dL[0])
    conf = SparkConf().setAppName("stroer poi")
    conf.setMaster('yarn')
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.executor.instances", "2")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    
    df = spark.read.parquet(baseDir + 'raw/others/poi_realtime/output/2019/05/29/00/30')

if False:
    from pyspark.sql import SparkSession, Row
    from pyspark.conf import SparkConf
    from pyspark.sql.functions import udf, struct, col, lit
    from functools import reduce
    from operator import add
    conf = SparkConf().setAppName("poi realtime")
    conf.setMaster('yarn')
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.executor.instances", "2")
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    import subprocess
    patterN = "part-00"
    cmd = 'hdfs dfs -ls /tmp/poi_realtime/output/2019/05/29/*/*/*'+patterN+'*'
    files = str(subprocess.check_output(cmd,shell=True))
    dL = [x.split(" ")[-1] for x in files.strip().split('\\n')]
    dL = list(set(['/'.join(x.split('/')[:-1]) for x in dL]))
    df = spark.read.parquet(dL[0])
    def flatten(d):
        g = dict(d['metrics']['crm'],**d['metrics']['crm_cross'])
        g['time'] = d['date']
        g['id_poi'] = d[idField]
        return Row(**g)

    d = df.take(1)[0]
    flatten(d)
    dirT = dirT.rdd.map(flatten).toDF()

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')

