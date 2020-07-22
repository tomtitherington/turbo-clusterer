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

def get_cluster_summary(store, run, cluster_number):
    cluster = store.select('clusters/r{}'.format(run), where='cluster == cluster_number')
    return cluster['radius'], cluster['centroid_0'], cluster['centroid_1']

"""
Returns a sample of stop points less than or equal to the number of samples specified.
"""
def sample(store, n_samples, day):
    samples = []
    nrows = store.get_storer('sp/d{}'.format(day)).nrows
    r = np.random.randint(0, nrows, size=n_samples)
    samples = store.select('sp/d{}'.format(day), where=pd.Index(r))
    return samples

"""
Returns a value for K using the elbow method on a random sample of n_samples.
"""
def sample_elbow(store, n_samples=10000, day=None):
    sample_set = sample(store, n_samples, day)
    long_lats = sample_set.filter(items=['longitude', 'latitude'])
    model = KMeans()
    visualiser = KElbowVisualizer(model, k=(1, 12))
    visualiser.fit(long_lats)
    return visualiser.elbow_value_


"""
Runs the elbow method multiple times each on a sample of random the points, returning the most occuring value of k.
"""
def elbow_mode(store,day=None):
    elbow_values = []
    for i in range(0, max_iter):
        elbow_values.append(sample_elbow(store,day=day))
    counter = Counter(elbow_values)
    mode = counter.most_common(1)[0][0]
    return mode
