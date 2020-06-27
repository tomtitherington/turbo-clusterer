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
    print(sp.get_stops(df, len(df.index), 50, 3))
    print(sp.get_stops(df, len(df.index), 100, 3))


def cluster_sp(store, order, threshold):
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp', chunksize=10000):
        for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store)


def birch_compare(store, order, threshold):
    # sklearn birch
    set = store.select('sp')
    df = set.filter(items=['longitude', 'latitude'])
    brc = Birch(threshold=threshold, branching_factor=order, n_clusters=None)
    sk_birch_centers = brc.fit(df).subcluster_centers_
    #print("sk birch:")
    # print(sk_birch_centers)
    # cluster feature tree
    tree = cft.CFTree(order, threshold)
    for _, row in df.iterrows():
        tree.insert_point(row.values)
    tree.save_tree(store)

    leaf_layer = clusters_at_max_height(store)
    x = leaf_layer['centroid_0'].tolist()
    y = leaf_layer['centroid_1'].tolist()
    #cft_centers = list(zip(x,y))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(*zip(*sk_birch_centers), c='b', marker="s", label='skbirch')
    ax1.scatter(x, y, c='r', marker="o", label='cftree')
    plt.legend(loc='upper left')
    plt.show()
    #print("cf tree:")
    # print(leaf_layer)


def clusters_at_max_height(store):
    df = store.select('clusters')
    max = df['layer'].max()
    return df[df.layer == max]


def get_leaf_centers(store, run):
    df = store.select('clusters/r{}'.format(run))
    pd.set_option('display.max_rows', df.shape[0] + 1)
    print(df)
    return df['layer'].max()


def re_run(store, cluster_number, order, threshold, run):
    print(store.get_storer('sp').nrows)
    cluster = store.select('clusters', where='cluster == cluster_number')
    radius = cluster['radius']
    centroid_long = cluster['centroid_0']
    centroid_lat = cluster['centroid_1']
    print(centroid_lat)
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp', chunksize=10000):
        # where='(longitude - centroid_long)**2 <= radius & (latitude - centroid_lat)**2 <= radius'
        chunk_filter = chunk.eval('(longitude - @centroid_long)**2 <= @radius & (latitude - @centroid_lat)**2 <= @radius', inplace=False)
        # chunk_filter = chunk[(chunk['longitude'] - centroid_long)**2 <= radius ]
        print(chunk_filter)
        #for _, row in (chunk[['longitude', 'latitude']]).iterrows():
            #print('working')
            #tree.insert_point(row.values)
    #tree.save_tree(store, run)


def plot_centers(store, centroids_layer, run):
    clusters = store.select('clusters/r{}'.format(run),
                            where='layer == centroids_layer')
    print(clusters)
    cluster_filter = clusters.filter(
        items=['centroid_0', 'centroid_1', 'n', 'radius'])
    cluster_sum = cluster_filter[cluster_filter['n'] > 100]

    areas = []
    scalar = 10
    for _, row in cluster_sum.iterrows():
        areas.append(np.pi * row['radius']**2 * scalar)
    cluster_sum['area'] = np.array(areas)

    # for printing
    pd.set_option('display.max_rows', cluster_sum.shape[0] + 1)
    print(cluster_sum)
    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
    # change the radius to be the area of the cluster
    ax = sns.scatterplot(x="centroid_0", y="centroid_1",
                         hue="n", size="area",
                         palette=cmap,
                         data=cluster_sum)
    ax.set(xlabel='Longitude', ylabel='Latitude')
    #plt.legend(title='Cluster',labels=['Number of clusters', 'Area'])
    # size should use the actual radius
    # colour should be n (darker the more points)
    # plt.scatter(cluster_sum['centroid_0'], cluster_sum['centroid_1'], s=cluster_sum['n'],
    #             c=cluster_sum['n'], cmap="Blues", alpha=0.4, edgecolors="grey", linewidth=2)
    plt.show()


# stop_point_test(store)

# print(sample_elbow(store))
# print(elbow_mode(store))

# threshold = k_means_thresh(elbow_mode(store),store)
# threhold = k_means_thresh(3,store)
# print("heuristic threshold = {}".format(threshold))
#
# df = sample(store,10000)
# long_lats = df.filter(items=['longitude','latitude'])
#

#print(get_leaf_centers(store,2))
plot_centers(store, 0,2)

threshold = 0.12203481466857011

# clusters = store.select('clusters', where='layer == 1')
# print(clusters[clusters.n > 10000])


#birch_compare(store, 50, threshold)

# cluster_sp(store, 50, threshold)
# print(get_leaf_centers(store))

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
