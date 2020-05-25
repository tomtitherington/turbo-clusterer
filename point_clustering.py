#!/usr/bin/env python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import Birch, KMeans

import stop_point_detection as spd
import cf_tree as cft



def build_tree(df, order, threshold, layer):
    tree = cft.CFTree(order,threshold)
    lnglats = df[['longitude', 'latitude']]
    count = 0
    for index, row in lnglats.iterrows():
        tree.insert_point(row.values)
        count+=1
    print("Count {}".format(count))
    tree.show()
    return tree



def birch(df):
    # threhold must be lowered (default=0.5)
    brc = Birch(threshold=0.05, n_clusters=None)
    brc.fit(df)
    return brc.predict(df)


def kmeans(df):
    kmeans = KMeans(n_clusters=2).fit(df)
    return kmeans.labels_


def plot(data, labels):
    plt.figure()
    plt.scatter(data['longitude'], data['latitude'], c=labels, cmap='rainbow')


def cluster(df, type, show_plot=None):
    lnglats = df[['longitude', 'latitude']]
    if type == 'birch':
        labels = birch(lnglats)
    elif type == 'kmeans':
        labels = kmeans(lnglats)

    if show_plot is not None:
        plot(lnglats, labels)
        plt.show()
    return labels


# read each taxis points and add them into the cf tree
# do this incrementally for each taxi

# lnglats = sp[['longitude', 'latitude']]
# lnglats100 = sp100[['longitude', 'latitude']]
# print(birch(lnglats))
# print(kmeans(lnglats))
#
# # plot(lnglats,kmeans(lnglats))
# plot(lnglats, birch(lnglats))
#
# plt.show()
