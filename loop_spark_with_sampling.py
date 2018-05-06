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
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform


# iteration used for each data point to get its
# 1. probablilistic set distance
# 2. its k-nearest neighbors
def loop_single_object(data, i, k, extent):
    
    all_dis = data[i]
    all_dis_index_sorted = np.argsort(all_dis)
    k_neighbor_points_index = all_dis_index_sorted[1:k+1]
    k_neighbor_sum = np.sum(all_dis[all_dis_index_sorted][1:k+1])
    
    #standard distance
    stand_diff = np.sqrt(k_neighbor_sum / k)
    #probablilistic set distance
    prob_set_diff = extent * stand_diff
    return prob_set_diff, k_neighbor_points_index

# used to calculate the probablilistic set distance for a single data point s in the context set of o (s belongs to S(o))
def loop_single_S(data, i, k, extent):
    
    all_dis = data[i]
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

# get the loop square for each data point
def loop_sampling(data, k=20, extent=2, sample=0.3):
    result = []
    PLOF_list = []
    num_sample = len(data)
    sample_size = int(num_sample * sample)
    for i in range(num_sample):
        data_new = np.delete(data,i,0)
        sampling_index = np.random.permutation((num_sample-1))[:sample_size]
        data_sample = data_new[sampling_index]
        #append the current data point to randomly sampled data points
        data_sample_use = np.append(data_sample, [data[i]], axis=0)
        
        # get pairwise distance squared among all the data points
        data_distance = squareform(pdist(data_sample_use))**2
        prob_set_diff, k_neighbor_points_index = loop_single_object(data_distance, sample_size,k ,extent)
        expected_pst = ev_sd(data_distance, k_neighbor_points_index, k,extent)
        if (expected_pst == 0):
            PLOF = 0
            PLOF_list.append(PLOF)
        else:
            PLOF = (prob_set_diff / expected_pst) - 1.0
            PLOF_list.append(PLOF)
    PLOF_list = np.array(PLOF_list)
    nPLOF = extent * np.sqrt(np.mean(np.square(PLOF_list)))
    
    for j in range(len(data)):
        res = np.maximum(0, erf(PLOF_list[j] / (nPLOF * np.sqrt(2.0))))
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

# loop with k set to 20, and each time randomly select 30% of samples from the population
loop_score = loop(data_val, k=20, sample = 0.3)

nonna['loop_with_sampling'] = loop_score

# write to csv file
result = spark.createDataFrame(nonna)
result.coalesce(1).write.format('csv').options(header='true').save('result_loop_sampling.csv')

sc.stop()
