import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/')
baseDir = os.environ['LAV_DIR']
import pandas as pd
import numpy as np
import datetime
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from collections import Counter
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
from pyspark.sql.functions import collect_list
import geomadi.geo_octree as octree
import shapely as sh
import geopandas as gpd

sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
cred = json.load(open(baseDir + "credenza/geomadi.json"))

poi = pd.read_csv(baseDir + "raw/tank/poi.csv")
poi.loc[:,"octree"] = poi['octree'].apply(lambda x:str(x)[:9])

import importlib
importlib.reload(octree)

projDir = cred['hdfs']['address'] + "/tdg/data_dumps/tracebox_2nd/parquets/CALL_20180628_cutdown/part-r-00000-889fcfdc-aacb-4aaf-b236-94ab109798ec.gz.parquet"
traD = sqlContext.read.parquet(projDir)
traD = traD.select("M_TMSI","POS_FIRST_LON","POS_FIRST_LAT","START_TIME","POS_LAST_LON","POS_LAST_LAT","END_TIME","CALL_TYPE","CALL_STATUS","MOVING")
traD = traD.limit(100)

def get_octree(x,y):
    return encode(x,y,precision=13)
udf_octree = udf(get_octree,StringType())
def n_min(x): # 30 minutes
    return int(float(x)*0.016666666666666666*0.03333333333333333)
udf_min = udf(n_min,IntegerType())
udf_disp = udf(octree.calcVector,StructType([StructField("mod",FloatType(),False),
                     StructField("angle",FloatType(),False),
                     StructField("chi"  ,IntegerType(),False)]))
traD = traD.withColumn("ts",func.unix_timestamp('START_TIME','yyyy-MM-dd HH:MM:SS'))
traD = traD.withColumn("g",udf_octree('POS_FIRST_LON','POS_FIRST_LAT'))
if False:
    traD = traD.withColumn("ts2",func.unix_timestamp('END_TIME','yyyy-MM-dd HH:MM:SS'))
    traD = traD.withColumn("g2",udf_octree("POS_LAST_LON","POS_LAST_LAT"))
    traD = traD.withColumn("ds",traD.ts2-traD.ts1)
    traD = traD.select("M_TMSI","g1","ts1","g2","ts2","MOVING","CALL_STATUS")
traD = traD.select("M_TMSI","g","ts")
traD.show(5)

def keyVal(row):
    return row["M_TIMSI"], row.asDict()

traK = traD.rdd.map(keyVal)

def packF(x1,x2):
    return [x1,x2]

schema_count = ArrayType(StructType([
    StructField("g", StringType(), False),
    StructField("ts", IntegerType(), False)
]))
udf_pack = udf(packF,schema_count)

traD.withColumn("disp",udf_pack(traD["g"],traD["ts"])).show(5)

vector_udf = udf(lambda vector: sum(vector[3]), DoubleType())
df.withColumn('feature_sums', vector_udf(df.features)).first()

#traD.groupBy("M_TIMSI").agg()


count_udf = udf(
    lambda s: Counter(s).most_common(), 
    schema_count
)
(traD.groupBy("M_TMSI")
 .agg(collect_list("message").alias("message"))
 .withColumn("message", unpack_udf("message"))
 .withColumn("message", count_udf("message"))).show(5)

df = sc.parallelize([(10100720363468236,["what", "sad", "to", "me"]),
                     (10100720363468236,["what", "what", "does", "the"]),
                     (10100718890699676,["at", "the", "oecd", "with"])]).toDF(["id", "message"])
unpack_udf = udf(
    lambda l: [item for sublist in l for item in sublist]
)
schema_count = ArrayType(StructType([
    StructField("g", StringType(), False),
    StructField("ts", IntegerType(), False)
]))
count_udf = udf(
    lambda s: Counter(s).most_common(), 
    schema_count
)
from pyspark.sql.functions import collect_list
(df.groupBy("id")
 .agg(collect_list("message").alias("message"))
 .withColumn("message", unpack_udf("message"))
 .withColumn("message", count_udf("message"))).show(truncate = False)

















traD.select(udf_disp("g1","g2")).show(5)
traD.printSchema()

traG = pd.DataFrame(traD.collect(),columns=traD.schema.names)

traD.coalesce(10).write.mode("overwrite").format('com.databricks.spark.csv').save(baseDir + "log/ciccia.csv")

traG.to_csv(baseDir + "log/ciccia.csv",index=False)

traG = traG.sort_values('ds',ascending=False)
traG = traG[traG['g2'] != '']
traG = traG[traG['g1'] != '']
traG.loc[:,"dif"] = traG.apply(lambda x: (int(x['g1']) - int(x['g2'])),axis=1)
traG = traG.sort_values('dif',ascending=False)

import importlib
importlib.reload(octree)
importlib.reload(tlib)

traG.loc[:,"vec"] = traG.apply(lambda x:octree.calcVector(x['g1'],x['g2']),axis=1)
traG.loc[:,"vel"] = traG.apply(lambda x: x['vec'][0]/x['ds'],axis=1)
traG = traG.replace(float('inf'),0.)
traG = traG.replace(float('nan'),0.)
print("not null vel %.3f" % (traG[traG['vel'].abs() > 0].shape[0]/traG.shape[0]) )
limg = np.percentile(traG['vel'].abs(),[0.05,0.95,0.999999999999999999])
print(limg)
traT = traG[traG.vel.abs() > 0.]
limg = np.percentile(traT['vel'],[0.05,0.95])
traT = traT[traT.vel.abs() < limg[-1]]


print(octree.calcVector(traG.iloc[1]['g1'],traG.iloc[0]['g2']))

tlib.plotHist(traT['vel'],nBin=20,isLog=True)
tlib.plotHist(traT['vel'],nBin=20,isLog=False)


df1 = df.select('id_poi','M_TMSI',udf_octree('LON','LAT')
                ,func.when((df.DATE_TIME.isNull() | (df.DATE_TIME == '')),'0')\
                .otherwise(func.unix_timestamp('DATE_TIME','yyyy-MM-dd HH:MM:SS'))
                )
newColumns = ["id_poi","tmsi","octree","ts"]
for i,c in enumerate(df1.columns):
    df1 = df1.withColumnRenamed(c,newColumns[i])
df1 = df1.withColumn("count",func.lit(1))
df1 = df1.withColumn("ts",udf_min(df1.ts))
df1.show()

gf = df1.groupby(['id_poi','tmsi','octree','ts']).sum('count').withColumnRenamed('sum(count)','count')
pf = gf.groupby(['id_poi','octree']).sum('count').withColumnRenamed('sum(count)','count')
row_p = pf.filter(lambda x: [x.id_poi,x.count,octree.decodePoly(x.octree)]).collect()

df2 = df1.take(20)
p = sh.geometry.Polygon(octree.decodePoly(df2[0][2]))
poiG = gpd.GeoDataFrame( [['box', p]],
                     columns = ['shape_id', 'geometry'], 
                     geometry='geometry')

with open(baseDir+'gis/tank/trace_bbox.geojson','w') as f:
    f.write(poiG.to_json())

df = df.withColumn("count",func.lit(1))
gf = df.groupby(['id_poi','M_TMSI']).sum('count')
act = gf.toPandas()

act.columns = ["id_poi","id_user","count"]
gist = pd.read_csv(baseDir + "raw/tank/ref_year.csv.gz",compression="gzip")
gist.loc[:,"day"] = gist['time'].apply(lambda x:x[:10])
gist = gist[gist['day'] == "2018-06-28"]
del gist['time']
dist = gist.groupby(['day','id_poi']).agg(sum).reset_index()
def clampF(x):
    return pd.Series({"count":sum(x['count']),"len":len(x['count'])})
gact = act[['id_poi','count']].groupby('id_poi').apply(clampF).reset_index()
dist = pd.merge(dist,gact,on="id_poi",how="left")
dist = dist.replace(float('nan'),0)

print("%.2f - %.2f" % (sp.stats.pearsonr(dist['ref'],dist['count'])[0],sp.stats.pearsonr(dist['ref'],dist['len'])[0]) )

corL = []
for i in range(12):
    act1 = act[act['count'] > i]
    gact = act1[['id_poi','count']].groupby('id_poi').apply(clampF).reset_index()
    dist = gist.groupby(['day','id_poi']).agg(sum).reset_index()
    dist = pd.merge(dist,gact,on="id_poi",how="left")
    dist = dist.replace(float('nan'),0)
    corL.append({"n_event":i,"r_sum":sp.stats.pearsonr(dist['ref'],dist['count'])[0],"r_count":sp.stats.pearsonr(dist['ref'],dist['len'])[0]})
corL = pd.DataFrame(corL)

vact.loc[:,"geohash"] = vact[['x_n','y_n']].apply(lambda x: geohash.encode(x[0],x[1],precision=8),axis=1)
def clampF(x):
    return pd.Series({"n":sum(x['n']),"sr":np.mean(x['sr'])})
lact = vact.groupby('geohash').apply(clampF).reset_index()
for i in range(3):
    setL = lact['n'] < 30.
    lact.loc[:,"geohash2"] = lact['geohash']
    lact.loc[setL,"geohash"] = lact.loc[setL,'geohash2'].apply(lambda x: x[:(8-i-1)])
    lact = lact.groupby('geohash').apply(clampF).reset_index()
