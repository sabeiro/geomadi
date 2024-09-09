import pyspark
sc = pyspark.SparkContext.getOrCreate()
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt
from pyspark.sql.functions import to_utc_timestamp, from_utc_timestamp
from pyspark.sql.functions import date_format
from pyspark.sql import functions as func
from pyspark.sql.functions import col
sqlContext = SQLContext(sc)
import tarfile

d = '/home/sabeiro/lav/motion/log/statWeek/winter/mtc_day/20190219_1900_statWeek_thursday/2019/custom_aggregations/odm_result/custom_odm_result_application_1549372224025_0460/'
df = sqlContext.read.parquet(d)
