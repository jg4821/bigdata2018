#!/usr/bin/env python3

## ABOD algorithm
## author Jiayi Gao

import numpy as np
import pandas as pd
from datetime import datetime

from csv import reader
import csv
import sys
import string
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_string
from pyspark.sql.functions import date_format
from scipy.spatial import distance
from sklearn.preprocessing import scale


def dotABC(A,B,C):
	# features
	m = len(A)
	AC_vec= (C-A).reshape(m,1)
	AB_vec= (B-A).reshape(m,1)
	# Special notice
	#print(np.dot(AC_vec.T,AB_vec))
	return np.dot(AC_vec.T,AB_vec)

def ABOD_Scores(X,index,PairDistance):
	n_obs,_= X.shape
	scores = []    
	A = X[index,:]
	for i in range(n_obs):
		if index!=i:
			B = X[i,:]
			AB = PairDistance[i][index]	
			for j in range(i):
				if index !=i and j != i:
					C = X[j,:]
					AC =PairDistance[j][index]
				# One types of special considerations:
				# 1. AB or AC =0 which means that A==B or A==C?
					if AB!=0 and AC!=0:
						dotPro = dotABC(A,B,C)
						temp = dotPro / (AB*AC)**2
						scores.append(temp)
	res = np.var(scores)
	return res
	


sc = SparkContext()
spark = SparkSession.builder.appName('outlier test').config('spark.some.config.option','some-value').getOrCreate()

# reading data to DataFrame
data = spark.read.format('csv').options(header='true',inferschema='true').load(sys.argv[1])

pd_df = data.toPandas()

pd_df = pd_df.select_dtypes(include=['float64','int64'])

nonna = pd_df.dropna()

# Conduct data normalization with zero mean and unit variance

X_scaled = scale(nonna,axis=1)

n,_= X_scaled.shape

# compute the pairwise distances
# Please be careful the order of the output distance is (row0,row1),(row0,row2) and (row1,row2).

# time the computation
time = datetime.now()
PairDistance = distance.squareform(distance.pdist(X_scaled))
ABOD_res = []
for i in range(n):
	#cur_row = X_scaled[i,:]
	ABOD_res.append(ABOD_Scores(X_scaled,i,PairDistance))
diff_time = (datetime.now()-time).total_seconds()


# sort the ABOD_res result and give the rank for the top 100 points and their original index

index_sort = np.argsort(np.asarray(ABOD_res))

rank = np.empty(len(ABOD_res),dtype=int)
for i in range(len(ABOD_res)):
	r = int(i+1)
	ind = index_sort[i]
	rank[ind] = r

# append columns of lof score and running time to the dataset
nonna['abod'] = np.asarray(ABOD_res)
nonna['rank'] = rank
time_list = [diff_time] * nonna.shape[0]
time_list = np.asarray(time_list)
nonna['run_time(s)'] = time_list

# write to csv file
result = spark.createDataFrame(nonna)
result.coalesce(1).write.format('csv').options(header='true').save('abod_result.csv')

sc.stop()

