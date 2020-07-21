#!/usr/bin/env python3
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stop_point_detection as sp
import cf_tree as cft
import seaborn as sns
import sample as smpl
import mplleaflet


from sklearn.cluster import KMeans, Birch
from collections import Counter

"""
Clusters points that are within a specified geographical region.
"""
def refined_tree(store, order, threshold, run, radius, centroid_long, centroid_lat):
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp', chunksize=10000):
        chunk = chunk.query(
            'sqrt( (longitude - @centroid_long)**2 + (latitude - @centroid_lat)**2) < @radius')
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store, run)

"""
Clusters points per day using the structure of the HDF store
"""
def create_tree_per_day(store, order, threshold, run, day):
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp/d{}'.format(day), chunksize=10000):
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store, run)


def create_clusters(store, day, order, threshold):
    # Automatic threshold calculation
    if threshold is None:
        threshold = k_means_thresh(
            smpl.elbow_mode(store, day=day), store, day=day)

    # Saves CFT to store
    create_tree_per_day(store, order, threshold, order + day, day)

    # Plotting
    plt.clf()
    df = clusters_at_max_height(store, order + day)
    cluster_filter = df.filter(
        items=['centroid_0', 'centroid_1', 'n', 'radius'])
    cluster_filter = cluster_filter[cluster_filter['n'] > 1000]
    catchment = []
    for _, row in cluster_filter.iterrows():
        # radius in meters
        dist = sp.distance((row['centroid_0'], row['centroid_1']), (
            row['centroid_0'] + row['radius'], row['centroid_1'] + row['radius']))
        # area
        catchment.append(np.pi * dist**2)
    cluster_filter['size'] = catchment
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    plt.scatter(x="centroid_0", y="centroid_1", s="size", c="n",
                cmap='Reds', alpha=0.7, data=cluster_filter)
    plt.plot()
    mplleaflet.show()


"""
Calculates the threshold to use for birch clustering, based on the samples taken from the data.
"""
def k_means_thresh(mode, store, n_samples=1000, day=None):
    sample_set = smpl.sample(store, n_samples, day)
    long_lats = sample_set.filter(items=['longitude', 'latitude'])
    kmeans = KMeans(n_clusters=mode).fit(long_lats)
    avg_r = np.sqrt(kmeans.inertia_ / len(sample_set.index))
    return avg_r


def clusters_at_max_height(store, run):
    print("Run: {}".format(run))
    df = store.select('clusters/r{}'.format(run))
    max = df['layer'].max()
    pd.set_option('display.max_rows', df.shape[0] + 1)
    return df[df.layer == max]


def closest_cluster(clusters, long, lat):
    c_index = 0
    c_distance = sp.distance(
        (clusters.iat[0, 7], clusters.iat[0, 8]), (long, lat))
    index = 1
    for _, row in clusters.iloc[1:].iterrows():
        dist = sp.distance((row['centroid_0'], row['centroid_1']), (long, lat))
        if dist < c_distance:
            c_index = index
            c_distance = dist
        index += 1
    return clusters.iat[c_index, 0]


def distance_to_stop_plot(store, order, day):
    df = clusters_at_max_height(store, order + day)
    nrows = store.get_storer('sp/d{}'.format(day)).nrows
    r = np.random.randint(0, nrows, size=1000)
    samples = store.select('sp/d{}'.format(day), where=pd.Index(r))
    hist = []
    i = 0
    for _, row in (samples[['longitude', 'latitude']]).iterrows():
        dist = closest_cluster(df, row['longitude'], row['latitude'])
        print(i)
        i += 1
        hist.append(dist)
    plt.hist(x=hist, bins='auto', color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Distance (meters)')
    plt.ylabel('Frequency')
    plt.title('Distance to nearest bus stop')
    plt.show()
