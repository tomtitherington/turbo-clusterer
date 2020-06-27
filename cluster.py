#!/usr/bin/env python3
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stop_point_detection as sp
import cf_tree as cft
import seaborn as sns
import sample as smpl


from sklearn.cluster import KMeans, Birch
from collections import Counter

store = pd.HDFStore("taxi_store.h5")

### BIRCH clustering

def create_tree(store, order, threshold, run):
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp', chunksize=10000):
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store, run)

def refined_tree(store, order, threhold, run, cluster_number):
    tree = cft.CFTree(order, threshold)
    #radius, centroid_long, centroid_lat = smpl.get_cluster_summary(store, cluster_number)
    radius = 0.109577
    centroid_long = 116.385246
    centroid_lat = 39.921822
    for chunk in store.select('sp', chunksize=10000):
        chunk = chunk.query('sqrt( (longitude - @centroid_long)**2 + (latitude - @centroid_lat)**2) < @radius')
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store,run)


"""
Carries out clustering.
"""
def cluster(store, order, run):
    #threshold = k_means_thresh(smpl.elbow_mode(store),store)
    threshold = 0.12203481466857011
    create_tree(store, order, threshold, run)

### KMeans

"""
Calculates the threshold to use for birch clustering, based on the samples taken from the data.
"""
def k_means_thresh(mode, store, n_samples=1000, filter=None):
    sample_set = smpl.sample(store, n_samples,filter)
    long_lats = sample_set.filter(items=['longitude', 'latitude'])
    kmeans = KMeans(n_clusters=mode).fit(long_lats)
    total_d = 0
    for i in range(kmeans.cluster_centers_.shape[0] - 1):
        total_d += ((((kmeans.cluster_centers_[i + 1:] -
                       kmeans.cluster_centers_[i])**2).sum(1))**.5).sum()
    avg_d = total_d / \
        ((kmeans.cluster_centers_.shape[0] - 1)
         * (kmeans.cluster_centers_.shape[0]) / 2.)
    avg_r = np.sqrt(kmeans.inertia_ / len(sample_set.index))
    return 0.25 * avg_d + 0.9 * avg_r, avg_d, avg_r

#cluster 275
# df = store.select('clusters/1')
# max = df['n'].max()
# print(df[df.n == max])

#store.remove('clusters/')
#cluster(store,50,1)

#refined_tree(store, 50, )
threshold = 0.03499218719593619
refined_tree(store,50,threshold,2,275)
#print(k_means_thresh(4,store,filter=275))
