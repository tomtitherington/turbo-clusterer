#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
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


'''
    Description: loads the latitude and longitude points from the taxies in the
    specified range and plots them
    Params: lower [lower bound of taxi id] upper [upper bound of taxi id]
    type [determines which plot is rendered]
'''


def taxi_plot(lower, upper, type):
    # filenames = glob(
    #     'release/taxi_log_2008_by_id/[{0!s}-{1!s}].txt'.format(lower, upper))
    filenames = glob(
        'release/taxi_log_2008_by_id/*.txt')
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date/time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    #result_filtered = result[result['longitude'] >= 116]
    long_lat = result[['longitude', 'latitude']]

    if type == "scatter":
        sns.scatterplot(x="longitude", y="latitude", data=long_lat)
    elif type == "kde":
        sns.jointplot(x="longitude", y="latitude", data=long_lat,
                      kind="kde")
    # plt.show()


''' 02/02/2008 is a friday 08/02/2008'''


def distri_time_plot(lower, upper, start_date, end_date, num_dates):
    # filenames = glob(
    #     'release/taxi_log_2008_by_id/*.txt')
    filenames = os.listdir("/home/tithers/Documents/Computing Science/Thesis/turbo-clusterer/taxi_log_2008_by_id")
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date_time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    print(result)
    # Set up figure
    f = plt.subplots(1,num_dates, sharex=True, sharey=True)
    slice = result[(result['date_time'] >= '2008-02-0{0!s} 00:00:00'.format(start_date))]
    print(slice)







    # result_filtered = result[(result['date_time'] >= '2008-02-0{0!s} 00:00:00'.format(start_date)) & (
    #     result['date_time'] <= '2008-02-0{0!s} 23:59:59'.format(start_date))]
    # print(result_filtered[['longitude']])
    # sns.kdeplot(result_filtered[['longitude']], result_filtered[['latitude']], shade=True)
    # start_date+=1
    # print("here")
    # result_filtered = result[(result['date_time'] >= '2008-02-0{0!s} 00:00:00'.format(start_date)) & (
    #     result['date_time'] <= '2008-02-0{0!s} 23:59:59'.format(start_date))]
    # sns.kdeplot(result_filtered[['longitude']], result_filtered[['latitude']], shade=True)
    # start_date+=1
    # result_filtered = result[(result['date_time'] >= '2008-02-0{0!s} 00:00:00'.format(start_date)) & (
    #     result['date_time'] <= '2008-02-0{0!s} 23:59:59'.format(start_date))]
    # sns.kdeplot(result_filtered[['longitude']], result_filtered[['latitude']], shade=True)
    # f.tight_layout()
    #long_lat = result_filtered[['longitude', 'latitude']]

    # # Set up figure
    # f, axes = plt.subplots(1,num_dates, sharex=True, sharey=True)
    #
    # # Rotate the starting point around the cubehelix hue circle
    # for i in range(num_dates):
    #     print(i)
    #     # Create a cubehelix colormap to use with kdeplot
    #     # cmap = sns.cubehelix_palette(rot=-.4, light=1, as_cmap=True)
    #     # Filter
    #     date_filter = result_filtered[(result_filtered['date_time'] >= '2008-02-0{0!s} 00:00:00'.format(
    #         start_date)) & (result_filtered['date_time'] <= '2008-02-0{0!s} 23:59:59'.format(start_date))]
    #     print(date_filter)
    #     # Create plot
    #     sns.kdeplot(date_filter[['longitude']], date_filter[['latitude']], shade=True)
    #     start_date += 1
    #
    # f.tight_layout()


#distri_plot(0, 10357)
taxi_plot(0, 10357, "kde")

#distri_time_plot(0, 10357, 4, 6, 3)

plt.show()
