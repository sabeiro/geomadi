import os, sys, gzip, random, csv, json, datetime
import pandas as pd
import numpy as np
import pyspark
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import mean, min, max, sum
from pyspark.sql import functions as func
import findspark
from pyspark.sql import window
import re
baseDir = os.environ['LAV_DIR']

try:
    sc
except NameError:
    findspark.init()
    sc = pyspark.SparkContext('local[*]')
sqlContext = SQLContext(sc)

gradMeter = 111122.19769899677
dateL = os.listdir(baseDir + "log/jll2h/")

fileN = []
for i in dateL:
    df = sqlContext.read.parquet(baseDir + "log/jll2h/" + i)
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


fig = plt.figure()
plt.subplots_adjust(right=0.75)
ax1 = fig.add_subplot(111)
ax1.plot(fileN.stay,fileN.n,label="# cells")
#ax1.set_ylabel('counts')
ax2 = ax1.twinx()
ax2.plot(fileN.stay,fileN.N,'r-',label="# count")
#ax2.set_ylabel('counts', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
ax3 = ax2.twinx()
ax3.plot(fileN.stay,fileN.perc,'g-',label="loss")
ax3.set_ylabel('loss', color='g')
for tl in ax3.get_yticklabels():
    tl.set_color('g')
ax1.set_xlabel("timespan")
fig.legend()
plt.show()
        
if False:
    df.select("year", "model").save("newcars.csv", "com.databricks.spark.csv")
    win_spec=window.Window().partitionBy(['act_cell','hour'])
    df2 = ddf.withColumn('csum',func.sum(ddf.count).over(win_spec))
    ddf.show()
    ddf.registerTempTable("act_table")
    df2 = sqlContext.sql("""SELECT act_cell, hour, count, sum(count) OVER (ORDER BY act_cell, hour) as cumsum FROM act_table""")


