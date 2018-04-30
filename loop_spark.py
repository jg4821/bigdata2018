#!/usr/bin/env python3

import itertools
from math import erf
import numpy as np
import sys
import warnings
from csv import reader
import sys
import string
import pandas as pd

from csv import reader
import csv
import sys
import string
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_string
from pyspark.sql.functions import date_format


# calculate Euclidean distance
def distance(vector1, vector2):
    distance = np.sum((vector1 - vector2) ** 2)
    return distance

# iteration used for each data point to get its
# 1. probablilistic set distance
# 2. its k-nearest neighbors
def loop_single_object(data, single, k, extent):
    all_dis = []
    for i in range(len(data)):
        dis = distance(data[i], single)
        all_dis.append(dis)
    all_dis = np.array(all_dis)
    all_dis_index_sorted = np.argsort(all_dis)
    k_neighbor_points = data[all_dis_index_sorted][1:k+1]
    k_neighbor_sum = np.sum(all_dis[all_dis_index_sorted][1:k+1])

    #standard distance
    stand_diff = np.sqrt(k_neighbor_sum / k)
    #probablilistic set distance
    prob_set_diff = extent * stand_diff
    return prob_set_diff, k_neighbor_points

# used to calculate the probablilistic set distance for a single data point s in the context set of o (s belongs to S(o))
def loop_single_S(data, single, k, extent):
    all_dis = []
    for i in range(len(data)):
        dis = distance(data[i], single)
        all_dis.append(dis)
    all_dis = np.array(all_dis)
    all_dis_sorted = np.sort(all_dis)
    k_neighbor_sum = np.sum(all_dis_sorted[1:k+1])
    stand_diff = np.sqrt(k_neighbor_sum / k)
    prob_set_diff = extent * stand_diff
    return prob_set_diff

# Calcualte the expected value of probablilistic set distance for all objects in the context set of a data point
def ev_sd(data, k_neibor_point, k, extent):
    all_dis = []
    for point in k_neibor_point:
        dis = loop_single_S(data, point, k, extent)
        all_dis.append(dis)
    all_dis = np.array(all_dis)
    mean = np.mean(all_dis)
    return mean

# loop
def loop(data, k = 20, extent = 3):
    result = []
    PLOF_list = []
    for i in range(len(data)):
        prob_set_diff, k_neighbor_points = loop_single_object(data, data[i],k ,extent)
        expected_pst = ev_sd(data, k_neighbor_points, k,extent)
        PLOF = (prob_set_diff / expected_pst) - 1
        PLOF_list.append(PLOF)
    PLOF_list = np.array(PLOF_list)

    # nPLOF is the square root of the expected value of squared PLOF values in the PLOF_list
    nPLOF = extent * np.sqrt(np.mean(np.square(PLOF_list)))

    # get loop score
    for j in range(len(data)):
        res = np.maximum(0, erf(PLOF_list[j] / (nPLOF * np.sqrt(2))))
        result.append(res)
    return np.array(result)

sc = SparkContext()
spark = SparkSession.builder.appName('loop').config('spark.some.config.option','some-value').getOrCreate()

data = spark.read.format('csv').options(header='true',inferschema='true',delimiter='\t').load(sys.argv[1])

pd_df = data.toPandas()

# exclude non-numeric data
pd_df = pd_df.select_dtypes(include=['float64','int64'])

nonna = pd_df.dropna()
data_val = nonna.values

# loop with k set to 10
loop_score = loop(data_val, k=10)

nonna['loop'] = loop_score

# write to csv file
result = spark.createDataFrame(nonna)
result.coalesce(1).write.format('csv').options(header='true').save('result_loop.csv')

sc.stop()
