#!/usr/bin/env python

from csv import reader
import sys
import string
import csv
from pyspark import SparkContext


# check if the input value equals the null
def isNull(x):
	if x == null_val:
		return ('null',1)
	else:
		return ('null',0)


sc = SparkContext()

# get null value from user input 
null_val = sys.argv[2]

# read the data into a RDD and partition data with csv reader
data = sc.textFile(sys.argv[1], 1)
data = data.mapPartitions(lambda x: reader(x,delimiter='\t'))

# get number of rows and columns in this dataset
num_row = data.count() - 1
first = data.take(1)
num_col = len([item for x in first for item in x])

# count for each column the number of nulls and percentage of nulls
combined = sc.emptyRDD()
for i in range(num_col):
	col_nulls = data.map(lambda x: isNull(x[i]))
	count_null = col_nulls.reduceByKey(lambda x,y: x+y)
	count_null = count_null.mapValues(lambda v:'{0:s}, {1:d}, {2:.2f}'.format(first[0][i],v,float(v)/num_row))
	combined = sc.union([combined,count_null])

# print out the result
result = combined.map(lambda x:'{0:s}'.format(x[1]))
result.coalesce(1).saveAsTextFile('count_null.csv')

sc.stop()
