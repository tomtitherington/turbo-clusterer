#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tables
import os
import os.path
import argparse

import stop_point_detection as sp

store = pd.HDFStore("taxi_store.h5")

def clusters_at_max_height(store):
    df = store.select('clusters')
    max = df['layer'].max()
    return df[df.layer == max]

def find_cluster(clusters, long, lat):
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


def sp_assignment(store):
    try:
        clusters = clusters_at_max_height(store)
    except:
        print("No clusters were found")
        return
    for chunk in store.select('sp', chunksize=10000):
        chunk_clusters = []
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            cluster = find_cluster(clusters, row['longitude'], row['latitude'])
            chunk_clusters.append(cluster)
        print("chunk size: {}   chunk_cluster size: {}".format(len(chunk),len(chunk_clusters)))
        chunk['cluster'] = np.array(chunk_clusters)
        store.append("sp_cluster",chunk,format='table', index=['cluster'])


sp_assignment(store)
store.close()
