import sys
import csv
import json
import time
import datetime
from ast import literal_eval
import pandas as pd
import numpy as np
#import geohash
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
# from pyspark.sql.functions import udf
from pyspark.sql import HiveContext
from pyspark.sql import functions as func
import matplotlib.pyplot as plt
from pyspark.sql.functions import mean, min, max, sum
from pyspark.sql import functions as func
import glob
import matplotlib.colors
from collections import Counter
import pytz
# import natco_utils
# from spark_utils import create_spark_and_sql_context
#import act_config_utils as conf_util



fileL = ["HH-cilacs.csv","HH_day_tourists_intermediate_results","_IGNORED","activity_report_old","aggregated_events","aggregated_events_1000","aggregated_events_HH","aggregated_events_filtered","aggregated_events_filtered_berlin","aggregated_events_filtered_germany","aggregated_events_test","paths","stroer_results","subway_model","tdg","trips","trips_sample","trips_small_sample","whitelisted"] #hdfs dfs -ls /tdg/2017/04/11/


datapath="/media/sf_DData/"
figdir=datapath+ "ODM_results/JLL/plots/"
resultsdir=datapath+ "ODM_results/JLL/results/"
inputdir=datapath+ "ODM_results/JLL/input/"

time_mapping={2:"2",3:"3",4:"4",5:"5",6:"6",7:"7",8:"8",9:"9",10:"10",11:"11",12:"12",13:"13",14:"14",15:"15",\
             16:"16",17:"17",18:"18",19:"19",20:"20",21:"21",22:"22",23:"23",24:"0+1",25:"1+1"}
hours=[x for x in map(str,range(2,24))]+["0+1","1+1"]

#load mappingv

ms=pd.read_csv("/media/sf_DData/TDG_infrastructure/CI-LAC_MS.csv")



confusedRDD = sc.textFile("confused.txt")
confusedRDD.take(5)
mappedconfusion = confusedRDD.map(lambda line : line.split(" "))
mappedconfusion.take(5)
flatMappedConfusion = confusedRDD.flatMap(lambda line : line.split(" "))
flatMappedConfusion.take(5)
onlyconfusion = confusedRDD.filter(lambda line : ("confus" in line.lower()))
onlyconfusion.count()
onlyconfusion.collect()
changesRDD = sc.textFile("/opt/spark/CHANGES.txt")
Daschanges = changesRDD.filter (lambda line: "tathagata.das1565@gmail.com" in line)
Daschanges.count ()
ankurchanges = changesRDD.filter(lambda line : "ankurdave@gmail.com" in line)
ankurchanges.count()
sampledconfusion = confusedRDD.sample(True,0.5,3) //True implies withReplacement
sampledconfusion.collect()


abhay_marks = [("physics",85),("maths",75),("chemistry",95)]
ankur_marks = [("physics",65),("maths",45),("chemistry",85)]
abhay = sc.parallelize(abhay_marks)
ankur = sc.parallelize(ankur_marks)
abhay.union(ankur).collect()
Subject_wise_marks = abhay.join(ankur)
Subject_wise_marks.collect()
Cricket_team = ["sachin","abhay","michael","rahane","david","ross","raj","rahul","hussy","steven","sourav"]
Toppers = ["rahul","abhay","laxman","bill","steve"]
cricketRDD = sc.parallelize(Cricket_team)
toppersRDD = sc.parallelize(Toppers)
toppercricketers = cricketRDD.intersection(toppersRDD)
toppercricketers.collect()
best_story = ["movie1","movie3","movie7","movie5","movie8"]
best_direction = ["movie11","movie1","movie5","movie10","movie7"]
best_screenplay = ["movie10","movie4","movie6","movie7","movie3"]
story_rdd = sc.parallelize(best_story)
direction_rdd = sc.parallelize(best_direction)
screen_rdd = sc.parallelize(best_screenplay)
total_nomination_rdd = story_rdd.union(direction_rdd).union(screen_rdd)
total_nomination_rdd.collect()
unique_movies_rdd = total_nomination_rdd.distinct()
unique_movies_rdd .collect()


# from pyspark import SparkContext
# sc =SparkContext()

# mapping is in new format (as that in production)
# loaD MAPPING AS JSON

with open("./mltc_corrected.json") as data_file:
    mpjson = json.load(data_file)

# load targetcells
tc = pd.read_csv("./CI_LACS_JLL.csv", usecols=[10])
target_cells = tc["CI_LAC"].values

#helper functions
def get_ts(st):
    return map(lambda x: long(x.timestamp), st)

def get_label(x):
    return x["type"].keys()[0]

def get_duration(x):
    return x["duration"]
# obsolete
def get_dom_cell_from_steps(x):
    try:
        return x["dominant_cell"]
    except:
        return '-1'

def get_mapping_zone(cilac):
    try:
        mdict = mpjson["mapping"][cilac]
        zs = mdict.keys()
        ws = [mdict[zz]["weight"] for zz in zs]
        selected_zone = np.random.choice(zs, p=ws)
        return [str(selected_zone), str(mdict[selected_zone]["market_share"])]
    except:
        return['-1', '-1']


def prune_time_bins(binlist):
    bl = [int(bb) for b in binlist for bb in literal_eval(b)]
    bl = np.unique(bl).tolist()
    return bl


def filter_activity_by_location(cilac):
    return cilac in target_cells


# # version 2: only fixed hours
def calculate_time_bins_hourly(timestamps):
    start_bin = datetime.utcfromtimestamp(timestamps[0]).hour
    end_bin = datetime.utcfromtimestamp(timestamps[-1]).hour
    return range(start_bin, end_bin+1)

def get_age(crmfields):
    res = 0
    if "age" in crmfields:
        if crmfields["age"] is not None:
            res = crmfields["age"]
    return res


def get_gender(crmfields):
    res = 0
    if "gender" in crmfields:
        if crmfields["gender"] is not None:
            res = crmfields["gender"]
    return res


def get_mcc(otherfields):
    res = 0
    if "mcc" in otherfields:
        if otherfields["mcc"] is not None:
            res = otherfields["mcc"]
    return res


def main():
    configs=conf_util.get_all_confs()
    minimum_duration_lst = [1.0]

    #configs = config_utils.get_all_confs()
    fname = "tdg/2017/04/11/trips"
    #timezone = natco_utils.get_time_zone(configs)

    # Initialize SparkContext & SQLContext
    app_name = module_name = "activity_report"
    sc, sqlContext = create_spark_and_sql_context(configs, app_name, module_name)

    sqlContext = HiveContext(sc)

    udf_get_labeltype = udf(get_label, StringType())
    udf_get_duration = udf(get_duration, FloatType())
    udf_get_domcell = udf(get_dom_cell_from_steps, StringType())
    udf_get_timestamps = udf(get_ts, ArrayType(IntegerType()))
    # udf_get_bins = udf(calculate_time_bins_hourly, ArrayType(IntegerType()))
    udf_get_bins = udf(calculate_time_bins_hourly)
    udf_get_mapping = udf(get_mapping_zone, ArrayType(StringType()))
    udf_get_zone = udf(lambda x: x[0], StringType())
    udf_get_ms = udf(lambda x: x[1], StringType())
    # udf_prune_bins = udf(prune_time_bins, ArrayType(IntegerType()))
    udf_prune_bins = udf(prune_time_bins, ArrayType(IntegerType()))

    for minimum_duration in minimum_duration_lst:
        print "starting"
        df = sqlContext.read.parquet("/tdg/2017/04/11/trips")
        print "data loaded"
        df = df.filter((udf_get_labeltype("label") == "activity") & (udf_get_duration("metrics") >= minimum_duration))
        df = df.withColumn("act_cell", udf_get_domcell("section_extra_fields"))
        df = df.withColumn("ts", udf_get_timestamps("steps"))
        df = df.withColumn("timebins", udf_get_bins(df.ts))
        df = df.withColumn("mapping", udf_get_mapping("act_cell"))
        df = df.withColumn("market_share", udf_get_ms("mapping"))

        # TO DO later: get mcc. age, gender, carry along       
        # df = df.drop("steps")
        # df = df.drop("metrics")
        # df = df.drop("crm_fields")
        # df = df.drop("other_fields")
        # df = df.drop("label")
        # df = df.drop("act_cell")
        # df = df.drop("ts")
        # df = df.drop("original_chain_metrics")
        # df = df.drop("section_extra_fields")
        # df = df.drop("dominant_cell")

	df = df.select("imsi", "act_cell", "market_share", "timebins")

        df = df.groupby("imsi", "act_cell", "market_share").agg(func.collect_list("timebins"))
        df = df.withColumn("time_bins_pruned", udf_prune_bins("collect_list(timebins)"))
        df = df.withColumn("hour", func.explode(df.time_bins_pruned))
        df = df.select("act_cell", "hour","market_share")
        res = df.groupby("act_cell", "hour","market_share").count()

        res.coalesce(5).write.mode('overwrite').parquet("/tdg/2017/05/23/jll_activities_per_cell_" + str(minimum_duration))


if __name__ == '__main__':
    main()
