#!/usr/bin/env python

from csv import reader
import sys
import string
import csv
# import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_string
from pyspark.sql.functions import date_format



sc = SparkContext()
spark = SparkSession.builder.appName('clean data').config('spark.some.config.option','some-value').getOrCreate()

# reading data to DataFrame
data = spark.read.format('csv').options(header='true',inferschema='true',delimiter='\t').load(sys.argv[1])
null_val = int(sys.argv[2])

# convert spark dataframe to pandas dataframe
pd_df = data.toPandas()

row,col = pd_df.shape

# filter out any null values in the dataframe
pd_df = pd_df.dropna()

# filter out any non-typical null values
for i in list(pd_df.columns):
	pd_df = pd_df[pd_df[i]!=null_val]

result = spark.createDataFrame(pd_df)

# write cleaned data into csv file
result.coalesce(1).write.format('csv').options(header='true').save("clean_data.csv")


sc.stop()
