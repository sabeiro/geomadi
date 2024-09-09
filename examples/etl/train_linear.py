import findspark
findspark.init("/usr/local/spark")
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *

# Build the SparkSession
spark = SparkSession.builder \
   .master("local") \
   .appName("Linear Regression Model") \
   .config("spark.executor.memory", "1gb") \
   .getOrCreate()
   
sc = spark.sparkContext


rdd = sc.textFile('~/cal_housing.data')
rdd = rdd.map(lambda line: line.split(","))
df = rdd.map(lambda line: Row(longitude=line[0], 
                              latitude=line[1], 
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5], 
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()

header = sc.textFile('~/cal_housing.domain')

def convertColumn(df, names, newType):
  for name in names: 
     df = df.withColumn(name, df[name].cast(newType))
  return df 
columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome', 'population', 'totalBedRooms', 'totalRooms']
df = convertColumn(df, columns, FloatType())

df.groupBy("housingMedianAge").count().sort("housingMedianAge",ascending=False).show()

from pyspark.ml.linalg import DenseVector
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
df = spark.createDataFrame(input_data, ["label", "features"])

from pyspark.ml.feature import StandardScaler
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
scaler = standardScaler.fit(df)
scaled_df = scaler.transform(df)
scaled_df.take(2)
train_data, test_data = scaled_df.randomSplit([.8,.2],seed=1234)
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="label", maxIter=10, regParam=0.3, elasticNetParam=0.8)
linearModel = lr.fit(train_data)
predicted = linearModel.transform(test_data)
predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
labels = predicted.select("label").rdd.map(lambda x: x[0])
predictionAndLabel = predictions.zip(labels).collect()
predictionAndLabel[:5]
linearModel.coefficients
linearModel.intercept
linearModel.summary.rootMeanSquaredError
linearModel.summary.r2


