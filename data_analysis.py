#!/usr/bin/env python3
import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import stop_point_detection as sp

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans, Birch
from collections import Counter

#store = pd.HDFStore(filename)
max_iter = 100
elbow_values = []
store = pd.HDFStore("taxi_store.h5")


def sample(store, n_samples):
    samples = []

    nrows = store.get_storer('sp').nrows
    r = np.random.randint(0, nrows, size=n_samples)
    samples = store.select('sp', where=pd.Index(r))
    return samples


def sample_elbow(store, n_samples=1000):
    sample_set = sample(store, n_samples)
    print(sample_set)
    print(len(sample_set.index))
    long_lats = sample_set.filter(items=['longitude', 'latitude'])
    model = KMeans()
    # experiment with bounds of k
    visualiser = KElbowVisualizer(model, k=(1, 12))
    visualiser.fit(long_lats)
    print("elbow value: {}".format(visualiser.elbow_value_))
    return visualiser.elbow_value_


def elbow_hist(store):
    for i in range(0, max_iter):
        elbow_values.append(sample_elbow(store))

    plt.hist(elbow_values)
    plt.show()


def elbow_mode(store):
    for i in range(0, max_iter):
        elbow_values.append(sample_elbow(store))
    print("elbow values: {}".format(elbow_values))
    counter = Counter(elbow_values)
    mode = counter.most_common(1)[0][0]
    print("MODE: {}".format(mode))
    return mode


def k_means_thresh(mode, store, n_samples=1000):
    sample_set = sample(store, n_samples)
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


def birch(df, threshold=0.05):
    # threhold must be lowered (default=0.5)
    brc = Birch(threshold, n_clusters=None)
    brc.fit(df)
    return brc.predict(df)


def kmeans(df):
    kmeans = KMeans(n_clusters=2).fit(df)
    return kmeans.labels_


def plot(data, labels):
    plt.figure()
    plt.scatter(data['longitude'], data['latitude'], c=labels, cmap='rainbow')


def stop_point_test(store):
    df = store.get('logs/t{}'.format(207))
    print("average stops:")
    print(sp.detect(df, len(df.index), 200, 2))
    print("significant stops")
    print(len(df.index))
    # return
    print(sp.get_stops(df, len(df.index), 200, 2))


stop_point_test(store)

# print(sample_elbow(store))
# print(elbow_mode(store))

# threhold = k_means_thresh(elbow_mode(store),store)
# threhold = k_means_thresh(3,store)
# print("heuristic threshold = {}".format(threhold))
#
# df = sample(store,10000)
# long_lats = df.filter(items=['longitude','latitude'])
#
# plot(long_lats,birch(long_lats, threhold[0]))
# plot(long_lats,birch(long_lats, threhold[1]))
# plot(long_lats,birch(long_lats, threhold[2]))
# plt.show()

# elbow_hist(store)

# df = store.get('sp/t{}'.format(2475))
# print(df)
# visualiser.fit(long_lats)
# visualiser.show()

store.close()
