#!/usr/bin/env python3
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stop_point_detection as sp
import cf_tree as cft
import seaborn as sns


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, Birch
from collections import Counter

max_iter = 100
elbow_values = []
store = pd.HDFStore("taxi_store.h5")
#filter = '(longitude - @centroid_long)**2 <= @radius & (latitude - @centroid_lat)**2 <= @radius'

def get_cluster_summary(store, cluster_number):
    cluster = store.select('clusters/1', where='cluster == cluster_number')
    return cluster['radius'], cluster['centroid_0'], cluster['centroid_1']

"""
Returns a sample of stop points less than or equal to the number of samples specified.
"""
def sample(store, n_samples, cluster_number=None):
    samples = []
    nrows = store.get_storer('sp').nrows
    r = np.random.randint(0, nrows, size=n_samples)
    samples = store.select('sp', where=pd.Index(r))
    if cluster_number is None:
        return samples
    else:
        #radius, centroid_long, centroid_lat = get_cluster_summary(store,cluster_number)
        radius = 0.109577
        centroid_long = 116.385246
        centroid_lat = 39.921822
        #print("radius: {}, centroid_long: {}, centroid_lat: {}".format(radius, centroid_long, centroid_lat))
        #samples = samples[~samples.index.duplicated()]
        samples = samples.query('sqrt( (longitude - @centroid_long)**2 + (latitude - @centroid_lat)**2) < @radius')
        return samples
        # samples = samples[(samples.longitude - centroid_long)**2 <= radius ]
        # return samples


def sample_elbow(store, n_samples=10000, cluster_number=None):
    sample_set = sample(store, n_samples, cluster_number)
    #print(len(sample_set.index))
    long_lats = sample_set.filter(items=['longitude', 'latitude'])
    model = KMeans()
    visualiser = KElbowVisualizer(model, k=(1, 12))
    visualiser.fit(long_lats)
    print("elbow value: {}".format(visualiser.elbow_value_))
    return visualiser.elbow_value_


"""
Runs the elbow method multiple times each on a sample of random the points, returning the most occuring value of k.
"""
def elbow_mode(store, cluster_number=None):
    for i in range(0, max_iter):
        elbow_values.append(sample_elbow(store,cluster_number=cluster_number))
    print("elbow values: {}".format(elbow_values))
    counter = Counter(elbow_values)
    mode = counter.most_common(1)[0][0]
    print("MODE: {}".format(mode))
    return mode

#elbow_mode(store,275)
