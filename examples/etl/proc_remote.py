import os, sys, gzip, random, csv, json, re
import pandas as pd
import numpy as np
import datetime
import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import to_utc_timestamp,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func

sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
cred = json.load(open(baseDir + "credenza/geomadi.json"))

#sc.textFile("hdfs:///tdg/2017/04/13/tank_und_rast_activities_with_attributes_0_2.0")
projDir = "/tdg/2017/04/11/aggregated_events"
projDir = "/tdg/2017/04/12/trips/t_und_r_additional_metrics"
projDir = "/tdg/2017/04/12/tank_und_rast_activities_with_attributes_0_2.0"
projDir = "/tdg/2017/04/11/odm_agg/age_destination_gender_od_origin_time_destination_time_origin"
projDir = "/tdg/2017/04/11/aggregated_events_filtered_unibail"
projDir = "/tdg/2017/04/11/trips/odm/unibail/"
projDir = "/tdg/2017/04/11/activity_report/unibail/"
projDir = "/tdg/qsm/20180924_1120_activities_tr_tdgcilac_hours_de_2hmax_nok30_DOM_20170411/2017/04/11/activity_report/output/dominant_zone.previous_trip_chirality"
projDir = "/tdg/2018/custom_aggregations/odm_result/custom_odm_result_application_1538385106982_0139"

projDir = cred['hdfs']['address'] + "/home/hdfs/paras/tank_rast/join_poi_buf_250_generic_events"
df = sqlContext.read.parquet(projDir)

#df.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save("odm_cilac.csv")
df.show()
df.printSchema()
tripD = df.toPandas()
tripD.to_csv("odm_cilac.csv.tar.gz",compression="gzip",index=False)

for dd in df.take(1):
    print(dd.metrics)
    print(dd.labels)
    print(dd.route_info)
    print(dd.steps)
    print(dd.other_fields)
    print(dd.crm_fields)

if False:
    cd /opt/commons/run-scripts
    ./start_parquet_utils_yarn.sh tocsv "/tdg/2017/04/12/tank_und_rast_activities_with_attributes_0_2.0"  "/tdg/2017/04/12/tank_und_rast_activities_with_attributes_0_2.0_csv"

sc.addPyFile("/tmp/activity_report_util_scripts_2018_03_09_16_25_50.zip")
import act_config_utils as conf_util
import natco_utils
from spark_utils import create_spark_and_sql_context
import act_config_utils as conf_util
import sys
tc = pd.read_csv("/home/isturm/cilac_sel_tank.csv",header=None)
print '***'
print tc.columns
target_cells = tc[0].values

def cilac_is_in_target_list(cis,lacs):
    cilaclist=make_cilac(cis,lacs)
    res=[(xx in target_cells) for xx in cilaclist]
    return any(res)

def make_cilac(ci,lac):
    cilaclist=map(lambda x,y: str(x) +'-'+str(y),ci,lac)
    return cilaclist

# Initialize SparkContext & SQLContext
configs = conf_util.get_all_confs()
datestr = "2017/04/11"
df=sqlContext.read.parquet("/tdg/"+datestr+"/aggregated_events")
udf_filter_targetcilac = udf(lambda cis,lacs: cilac_is_in_target_list(cis,lacs), BooleanType())

print df.count()
df=df.filter(udf_filter_targetcilac('cell_id_arr','area_id_arr'))

df.write.parquet("/tdg/"+datestr+"/aggregated_events_filtered_tank_und_rast",mode="overwrite")
print df.count()

