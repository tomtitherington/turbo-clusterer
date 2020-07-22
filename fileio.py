#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tables
import os
import os.path
import argparse

import stop_point_detection as spd
import cf_tree as cft
import cluster


def connect_to_store(filename):
    return pd.HDFStore(filename)


def read_log(filename):
    df = pd.read_csv(filename, header=None, names=[
        "taxi_id", "date_time", "longitude", "latitude"])
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df


def initial_convert(store, data_path):
    """Converts logs from CSV(s) into a HDF5 file.

    Converts *all* log files found in the directory (none recursive), into a single HDF5 file
    under the group header logs/. The name of each log generated has the letter t followed by
    the name of the original file. For example: a log with filename '200.txt' can be found under
    the group '/logs/t200'.

    Args:
        filename: The name of the HDF5 file. If one does not exist, then it will be created.
        data_path: The path to the folder containing the taxi logs. Pass '.' if the files are
        located in the current working directory (relative to the script). Full path or relative path
        is accepted.
    Raises:
        ....
    """
    for name in os.listdir(data_path):
        fullpath = os.path.join(data_path, name)
        if os.path.isfile(fullpath):
            print("writing log: {}".format(os.path.splitext(name)[0]))
            store.append("logs/t{}".format(os.path.splitext(name)[0]), read_log(fullpath),
                         data_columns=['date_time'], format='table')


def store_sp_from_csv(store, n):
    for i in range(1, n):
        log = read_log("release/taxi_log_2008_by_id/{}.txt".format(i))
        # distance threshold = 50 meters - time threshold = 3 minutes
        sp = spd.detect(log, len(log.index), 50, 3)
        store.append("sp/t{}".format(i), sp)


def calculate_sp(store, n, delta_d=50, delta_t=3):
    """Calculates stop points from the logs.

    Calculates the stop points for the logs in the specified range. A stop point
    is created if a taxi has remained within a certain radius, within a specified
    period of time.

    Args:
        store: The data store.
        n: The number of taxies to generate the stop points for.
        delta_d: The distance threshold, in meters. Default is 50 metres.
        delta_t: The time threshold, in minutes. Default is 3 minutes.
    Raises:
        ....
    """
    avg_sps = 0
    for i in range(n[0], n[1]):
        try:
            log = store.get('logs/t{}'.format(i))
        except:
            print("File logs/t{} could not be opened".format(i))
            continue
        #sp = spd.detect(log, len(log.index), delta_d, delta_t)
        sp = spd.get_stops(log, len(log.index), delta_d, delta_t)
        avg_sps += len(sp.index)
        print("{} stop points generated for taxi {}".format(len(sp.index), i))
        store.append("sp/", sp, format='table', index=True)
    print("average number of stop points per log: {}".format(
        avg_sps / (n[1] - n[0])))


def split_sp(store, day):
    for chunk in store.select('sp', chunksize=10000):
        print(chunk)
        mask = (chunk['arrival_dt'] >= '2008-02-0{}'.format(day)
                ) & (chunk['departure_dt'] < '2008-02-0{}'.format(day + 1))
        chunk = chunk.loc[mask]
        store.append('sp/d{}'.format(day), chunk)


def day_split(store):
    for i in range(2, 9):
        split_sp(store, i)


def cluster_sp(store, day, order, threshold=None):
    cluster.create_clusters(store, day, order, threshold)

def get_sp(store, taxi):
    return store.get('sp/t{}'.format(taxi))


def read_clusters(store, layer):
    clusters = store.get('clusters/l{}'.format(0))
    print(clusters)


def readstore_log(store, taxi):
    log = store.get('logs/t{}'.format(taxi))
    print(log)


def delete_clusters(store):
    store.remove('clusters/')


def keys(store):
    print(store.keys())


def delete_sps(store):
    store.remove('sp/')


def delete_group(store, group):
    store.remove(group)


def delete_cluster_run(store, day, order):
    store.remove('clusters/r{}'.format(day + order))


ap = argparse.ArgumentParser()
# MOBAL
ap.add_argument('store', metavar='STORE', nargs=1, default='taxi_store.h5',
                help='the name of the HDF5 store file')
ap.add_argument('--convert', metavar='INDIR', nargs=1,
                help='the directory path of the csv file(s) to be converted and placed in the stores')
ap.add_argument('--spoints', metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'DISTANCE_THRESH', 'TIME_THRESH'), nargs=4,
                help='calculate stop points within the specified range, with time and distance threshold')
ap.add_argument('--sp_day_split', help='splits the stop points by day')
ap.add_argument('--auto_cluster', nargs=2, metavar=('DAY', 'ORDER'),
                help='cluster the stop points for the day and order specified with automatic threshold')
ap.add_argument('--cluster', nargs=3, metavar=('DAY', 'ORDER', 'THRESH'),
                help='cluster the stop points for the day, order and threshold specified')
ap.add_argument('--read_log', nargs=1, metavar=('TAXI_ID'))
ap.add_argument('--delete_run', nargs=2, metavar=('DAY', 'ORDER'),
                help='delete the clusters produced on a run on a specific day and order value')
ap.add_argument('--delete', choices=['logs', 'sp', 'clusters'],
                help='delete a group within the HDF5 store from the choices listed')
args = ap.parse_args()


store = connect_to_store(*args.store)
if args.convert:
    initial_convert(
        store, *args.convert)  # initial convertion
if args.delete_run:
    delete_cluster_run(store, args.delete_run[0], args.delete_run[1])
if args.delete:
    delete_group(store, args.delete)
if args.spoints:
    calculate_sp(store, (int(args.spoints[0]), int(args.spoints[1])), float(
        args.spoints[2]), float(args.spoints[3]))  # stop point calculation
if args.sp_day_split:
    day_split(store)
if args.auto_cluster:
    cluster_sp(store, int(args.auto_cluster[0]), int(
        args.auto_cluster[1]))  # auto stop point clustering
if args.cluster:
    cluster_sp(store, int(
        args.cluster[0]), int(args.cluster[1]), float(args.cluster[2]))  # stop point clustering
if args.read_log:
    readstore_log(store, int(*args.read_log))


store.close()
