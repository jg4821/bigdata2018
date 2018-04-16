#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from datetime import datetime

from csv import reader
import csv
import sys
import string
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_string
from pyspark.sql.functions import date_format


class Lof():
	def __init__(self, k):
		self.k = k
	def fit(self, data):
		# compute distance between each pair of points in the dataset
		distance = squareform(pdist(data))
		indices = stats.mstats.rankdata(distance, axis=1)
		indices_k = indices <= self.k
		
		# compute k distance of each point
		kdist = np.zeros(len(data))
		for i in range(data.shape[0]):
			kneighbours = distance[i, indices_k[i, :]]
			kdist[i] = kneighbours.max()

		# compute the local reachability distance
		lrd = np.zeros(len(data))
		for i in range(data.shape[0]):
			# reachability distance of k nearest points
			# is the max of kdist of the point or the actual distance
			lrd[i] = 1/np.maximum(kdist[indices_k[i, :]], distance[i, indices_k[i, :]]).mean()
		
		# compute lof
		lof = np.zeros(len(data))
		for i in range(data.shape[0]):
			lof[i] = lrd[indices_k[i, :]].mean()/lrd[i]
		return lof


sc = SparkContext()
spark = SparkSession.builder.appName('outlier test').config('spark.some.config.option','some-value').getOrCreate()

# reading data to DataFrame
data = spark.read.format('csv').options(header='true',inferschema='true',delimiter='\t').load(sys.argv[1])

pd_df = data.toPandas()

# exclude non-numeric data
pd_df = pd_df.select_dtypes(include=['float64','int64'])

nonna = pd_df.dropna()

# initialize the lof model with k
model = Lof(k=10)

time = datetime.now()

lof = model.fit(nonna)

diff_time = (datetime.now()-time).total_seconds()


sc.stop()
