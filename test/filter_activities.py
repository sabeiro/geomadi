import pytz
from datetime import datetime
from ast import literal_eval
import pandas as pd
import numpy as np
import json, sys
from pyspark.sql import HiveContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql import functions as func
from collections import Counter
celllist = "cilac_sel_roda.csv"
tc = pd.read_csv("cilac_sel/" + celllist,header=None)
target_cilacs = tc[0].values

#configs = conf_util.get_all_confs()
minimum_duration_lst = [0]
maximum_duration_lst = [999.0]
minimum_duration = minimum_duration_lst[0]
maximum_duration = maximum_duration_lst[0]
datestr="2017/04/11"
fname = "/tdg/"+datestr+"/trips/unibail/work_feature"
app_name = module_name = "activity_report"
try:
    sc, sqlContext = create_spark_and_sql_context(configs, app_name, module_name)
except:
    print("")

sc.setLogLevel("ERROR")
sqlContext = HiveContext(sc)

df = sqlContext.read.parquet(fname)
def get_label(lab):
    return(lab["type"].keys()[0])

udf_get_label=udf(get_label,StringType())
df = df.filter((udf_get_label("labels") == "activity") )#& (df.metrics['duration'] >= minimum_duration) & (df.metrics['duration'] <= maximum_duration) )
df = df.withColumn("cilac",df.metrics['dominant_cell'])
df = df.where(df.cilac.isin(list(target_cilacs)))
df = df.withColumn("dist",df.metrics['distance'])
df = df.withColumn("dur",df.metrics['duration'])

def get_val(steps):
    xV = [float(x['x']) for x in steps if not x['x'] == "None"]
    return float(sum(xV)/len(xV))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("x",udf_get_val("steps"))

def get_val(steps):
    xV = [float(x['x']) for x in steps if not x['x'] == "None"]
    return float(np.std(xV)/np.sqrt(len(xV)))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("sx",udf_get_val("steps"))
    
def get_val(steps):
    xV = [float(x['y']) for x in steps if not x['y'] == "None"]
    return float(np.average(xV))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("y",udf_get_val("steps"))
    
def get_val(steps):
    xV = [float(x['y']) for x in steps if not x['y'] == "None"]
    return float(np.std(xV)/np.sqrt(len(xV)))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("sy",udf_get_val("steps"))

def get_val(steps):
    xV = [float(x['timestamp']) for x in steps if not x['timestamp'] == "None"]
    return float(np.average(xV))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("t",udf_get_val("steps"))
    
def get_val(steps):
    xV = [float(x['timestamp']) for x in steps if not x['timestamp'] == "None"]
    return float(np.std(xV)/np.sqrt(len(xV)))

udf_get_val = udf(get_val,FloatType())
df = df.withColumn("st",udf_get_val("steps"))
    
if False:
    df.select(df.steps.getItem('x')   ).show()
    df.select(df.steps.agg(sum("x"))).show()
    maxLength = df.select(size('steps').as("l")).groupBy().max("l").first.getInt(0)

    df.withColumn("x",explode($"emp_details"))
    .groupBy("dept_nm")
    .agg(sum("emp_sum")).show)


# df = df.withColumn("dist_prev_next",df.metrics['neighbouring_trips_dist_sum'])
# df = df.withColumn("dur_prev_next" ,df.metrics['neighbouring_trips_durs_sum'])
# df = df.withColumn("cilac_prev",df.metrics['chain_first_cilac'])
# df = df.withColumn("cilac_next",df.metrics['chain_last_cilac'])
df = df.withColumn("tac",df.other_fields['tac'])
df = df.withColumn("mcc",df.other_fields['mcc'])
df = df.withColumn("age",df.crm_fields['age'])
df = df.withColumn("gender",df.crm_fields['gender'])
if False:
    udf_get_mapping = udf(get_mapping_zone, ArrayType(StringType()))
    df = df.withColumn("mapping", udf_get_mapping("domcell"))
    udf_get_ms = udf(lambda x: x[1], StringType())
    df = df.withColumn("market_share", udf_get_ms("mapping"))            

for d in ["crm_fields","other_fields","labels","metrics","route_info","steps"]:
    df = df.drop(d)

print(df.columns)
if False:
    df.write.mode('overwrite').parquet("/tdg/"+datestr+"/activity_report/unibail")# + str(minimum_duration)+'_'+ str(maximum_duration))

zipD = df.toPandas()
zipD.to_csv("act_roda_13.csv.tar.gz",compression="gzip",index=False)
#df.coalesce(1).write.mode("overwrite").format('com.databricks.spark.csv').save("act_unibail.csv")

print('-----------------te-se-qe-te-ve-be-te-ne------------------------')
