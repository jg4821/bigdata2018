#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:53:06 2018

@author: tinghaoli
"""

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.decomposition import PCA
import os

DataPath = "/Users/tinghao/Desktop/SelectedDataColumns/clean"

os.chdir(DataPath)

data_path = "nbgq-j9jt.csv"
# choose pca or tsne to shrink the dimension
choice = 'tsne'

# Reading data
data = pd.read_csv(data_path)

# select the X data
X = np.array(data.iloc[:,:-5])
# select the y, it could be LOF, LoOP, ABOD, ABOD/LoOP with sampling as well
y = np.array(data.iloc[:,-3])

n=len(y)
# Data Preprocessiong

scaler=preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X) 


# Data transformation, namely shrink the Xs
# T-SNE
if choice == "tsne":
    X_trans = TSNE(n_components=2).fit_transform(X_scaled)
# PCA
else:
    pca = PCA(n_components=2)
    X_trans = pca.fit(X_scaled).transform(X_scaled)



# Preparation done, plot the results
fig, ax = plt.subplots(figsize=(6, 6))
legend_inx=np.argmax(y)

# For ABOD, or rank typed results
# thresholds to show the outliers. If larger than the thresholds, its outlier scores
# will be shown in the graph
threshold = 10
for i, txt in enumerate(y):
    # Lable only the first outlier index to aviod duplicates
    # 's' represents the areas of the points in the plot
    if n-y[i]< threshold:
        ax.scatter(X_trans[i,0],X_trans[i,1],c='red',s=20,label="Potential Outlier" if i==legend_inx else "" )
        ax.annotate(str(round(n-y[i]+1, 2)), (X_trans[i,0],X_trans[i,1]))
    else:
        ax.scatter(X_trans[i,0],X_trans[i,1],c='black',s=0.4)
        ax.annotate("", (X_trans[i,0],X_trans[i,1]))

ax.set_title('ABOD methods')
ax.set_title('ABOD with sampling methods 50%')
ax.legend(loc='best')



# Preparation done, plot the results
fig, ax = plt.subplots(figsize=(4, 4))
legend_inx=np.argmax(y)
# For LoF, LoOP, or socre typed results
# thresholds to show the outliers. If larger than the thresholds, its outlier scores
# will be shown in the graph
threshold =2
for i, txt in enumerate(y):
    # Lable only the first outlier index to aviod duplicates
    # 's' represents the areas of the points in the plot
    if y[i]>threshold:
        ax.scatter(X_trans[i,0],X_trans[i,1],c='red',s=y[i]*15,label="Potential Outlier" if i==legend_inx else "" )
        ax.annotate(str(round(y[i], 2)), (X_trans[i,0],X_trans[i,1]))
    else:
        ax.scatter(X_trans[i,0],X_trans[i,1],c='black',s=0.4)
        ax.annotate("", (X_trans[i,0],X_trans[i,1]))

ax.set_title('LoOP with sampling methods - 30%')
#ax.set_title('Original LoOP methods')
ax.set_title('LoF methods')

ax.legend(loc='best')




