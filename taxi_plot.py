#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


def taxi_plot():
    df = pd.read_csv('1.txt', header=None, names=[
                     "taxi_id", "date/time", "longitude", "latitude"])
    BBox = (df.longitude.min(), df.longitude.max(),
            df.latitude.min(), df.latitude.max())
    beij_map = plt.imread('beijing.png')

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(df.longitude, df.latitude, zorder=1, alpha=0.2, c='b', s=10)

    ax.set_title('Spatial Data of Taxi 1')
    ax.set_xlim(BBox[0], BBox[1])
    ax.set_ylim(BBox[2], BBox[3])

    ax.imshow(beij_map, zorder=0, extent=BBox, aspect='equal')

    plt.show()

# def taxi_plot(path):
# 	# path = 'release/taxi_log_2008_by_id/[{0!s}-{1!s}].txt'.format(lower, upper)
# 	filenames = glob(path)
#     dataframes = [pd.read_csv(f, header=None, names=["taxi_id", "date/time", "longitude", "latitude"]) for f in filenames]
#     result = pd.concat(dataframes)
#     result_filtered = result[result['longitude'] >= 116]
#     result_filtered.plot.scatter(x='longitude', y='latitude')
#     plt.show()


def taxi_plot(lower, upper):
    filenames = glob(
        'release/taxi_log_2008_by_id/[{0!s}-{1!s}].txt'.format(lower, upper))
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date/time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    result_filtered = result[result['longitude'] >= 116]
    result_filtered.plot.scatter(x='longitude', y='latitude')
    plt.show()


taxi_plot(0,10357)
# path = 'release/taxi_log_2008_by_id/[{0!s}-{1!s}].txt'.format(0, 10357)
# filenames = glob(path)
# print(path)
# # taxi_plot(path)
