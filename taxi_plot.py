#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import struct


def show_taxi_log():
    filenames = glob(
        'release/taxi_log_2008_by_id/*.txt')
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date_time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    print(result.head())

# Combine all taxi points and convert text files into a single hdf5 file
def csv_to_hdf():
    filenames = glob(
        'release/taxi_log_2008_by_id/*.txt')
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date_time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    # Writes to hdf5 format with taxi_log as the identifier for the group in the store
    # append=True
    # mode = 'w'
    result.to_hdf('taxi_log.h5', 'taxi_log', mode='w')


def read_taxi_hdf():
    # hdf = pd.HDFStore('taxi_log.h5', mode='r')
    # print(hdf['taxi_log'].head())
    # df = hdf.select('taxi_log')
    total_trajs = pd.read_hdf('taxi_log.h5', 'taxi_log')
    print(total_trajs.head())


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


def taxi_plot(type):
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
    filenames = os.listdir(
        "/home/tithers/Documents/Computing Science/Thesis/turbo-clusterer/taxi_log_2008_by_id")
    dataframes = [pd.read_csv(f, header=None, names=[
                              "taxi_id", "date_time", "longitude", "latitude"]) for f in filenames]
    result = pd.concat(dataframes)
    print(result)
    # Set up figure
    f = plt.subplots(1, num_dates, sharex=True, sharey=True)
    slice = result[(result['date_time'] >=
                    '2008-02-0{0!s} 00:00:00'.format(start_date))]
    print(slice)

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


#csv_to_hdf()
print(struct.calcsize("P") * 8)

#read_taxi_hdf()
# show_taxi_log()

#distri_plot(0, 10357)
# taxi_plot("kde")

#distri_time_plot(0, 10357, 4, 6, 3)

# plt.show()
