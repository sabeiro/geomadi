import os, sys, gzip, random, csv, json, re
sys.path.append(os.environ['LAV_DIR']+'/src/spark/')
baseDir = os.environ['LAV_DIR']
#sambaDir = os.environ['SAMBA_PATH']
sambaDir = "/run/user/1000/gvfs/smb-share:server=192.168.2.254,share=motionlogic/Data_Science/Customer_Projects_DE/"
import pandas as pd
import numpy as np
import datetime
import proc_lib as plib
import findspark
findspark.init()
import pyspark
#sc = pyspark.SparkContext('local[*]')
sc = pyspark.SparkContext.getOrCreate()

from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_utc_timestamp,from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
sqlContext = SQLContext(sc)
import pymongo
gradMeter = 111122.19769899677

key_file = baseDir + '/credenza/geomadi.json'
cred = []
with open(key_file) as f:
    cred = json.load(f)

sc.textFile("hdfs:///tdg/2017/04/13/tank_und_rast_activities_with_attributes_0_2.0")
