#!/usr/bin/python3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.cluster import Birch, KMeans

import stop_point_detection as spd


def birch(df):
    # threhold must be lowered (default=0.5)
    brc = Birch(threshold=0.05,n_clusters=None)
    brc.fit(df)
    return brc.predict(df)

def kmeans(df):
    kmeans = KMeans(n_clusters=2).fit(df)
    return kmeans.labels_

def plot(data,labels):
    plt.figure()
    plt.scatter(data['longitude'],data['latitude'],c=labels,cmap='rainbow')



df = pd.read_csv('release/taxi_log_2008_by_id/1.txt', header=None, names=[
                 "taxi_id", "date_time", "longitude", "latitude"])
df['date_time'] = pd.to_datetime(df['date_time'])
# distnace threshold 50 meters, time threshold 3 minutes
sp = spd.detect(df, len(df.index), 50, 3)
sp100 = spd.detect(df, len(df.index), 100, 3)

lnglats = sp[['longitude','latitude']]
lnglats100 = sp100[['longitude','latitude']]
print(birch(lnglats))
print(kmeans(lnglats))

#plot(lnglats,kmeans(lnglats))
plot(lnglats,birch(lnglats))

plt.show()
