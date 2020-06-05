#!/usr/bin/env python3
import pandas as pd
import numpy as np
import tables
import os
import os.path
import argparse

import stop_point_detection as spd
import cf_tree as cft


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
        #store.append("sp/t{}".format(i), sp, format='table', index=False)
        store.append("sp/", sp, format='table', index=False)
    print("average number of stop points per log: {}".format(
        avg_sps / (n[1] - n[0])))


def cluster_sp(store, order, threshold, r):
    tree = cft.CFTree(order, threshold)
    for chunk in store.select('sp', chunksize=10000):
        for _, row in (chunk[['longitude','latitude']]).iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store)

    # # read each taxis stop points in range r
    # for i in range(r[0], r[1]):
    #     try:
    #         df = store.get('sp/t{}'.format(i))
    #     except:
    #         print("File sp/t{} could not be opened".format(i))
    #         continue
    #     lnglats = df[['longitude', 'latitude']]
    #     for index, row in lnglats.iterrows():
    #         tree.insert_point(row.values)
    # tree.save_tree(store)

# "cluster", "layer", "n", "ls_0", "ls_1", "ss", "radius", "centroid_0", "centroid_1"


def distance(c1, c2):
    x = c1[0] - c2[0]
    y = c1[1] - c2[1]
    x *= x
    y *= y
    return np.sqrt(x + y)

# TODO: must change to use the syntax of a data frame, see point clustering file


def find_cluster(clusters, long, lat):
    print(long, lat)
    c_index = 0
    c_distance = distance(
        (clusters.iat[0, 7], clusters.iat[0, 8]), (long, lat))
    index = 1
    for _, row in clusters.iloc[1:].iterrows():
        dist = distance((row['centroid_0'], row['centroid_1']), (long, lat))
        if dist < c_distance:
            c_index = index
            c_distance = dist
        index += 1
    return clusters.iat[c_index, 0]


def create_cluster_sequence(store, taxi, layer):
    try:
        clusters = store.get('clusters/l{}'.format(layer))
        taxi_sp = store.get('sp/t{}'.format(taxi))
    except:
        print("Cluster sequence at layer {} could not be calculated".format(layer))
        return
    print(clusters)
    cluster_seq = np.array([])
    for _, row in taxi_sp.iterrows():
        cluster = find_cluster(clusters, row['longitude'], row['latitude'])
        #print('cluster id: {}'.format(cluster))
        cluster_seq = np.append(cluster_seq, cluster)
    taxi_sp['cluster'] = cluster_seq
    print(cluster_seq)
    store.append("sp/t{}".format(taxi), taxi_sp,  format='table',
                 append=False, index=False)


def create_cluster_sequences(store, r, layer):
    for taxi in (r[0], r[1]):
        create_cluster_sequence(store, taxi, layer)


def read_clusterseq(store, taxi):
    df = store.get('sp/t{}'.format(taxi))
    for _, row in df.iterrows():
        print(row)
        print("\n")


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


ap = argparse.ArgumentParser()
# MOBAL
ap.add_argument('store', metavar='STORE', nargs=1, default='taxi_store.h5',
                help='the name of the HDF5 store file')
ap.add_argument('--convert', metavar='INDIR', nargs=1,
                help='the directory path of the csv file(s) to be converted and placed in the stores')
ap.add_argument('--spoints', metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'DISTANCE_THRESH', 'TIME_THRESH'), nargs=4,
                help='calculate stop points within the specified range, with time and distance threshold')
ap.add_argument('--cluster', nargs=4, metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'ORDER', 'THRESH'),
                help='cluster the stop points of taxis in a specified range with branching factor and threshold')
ap.add_argument('--clusterseq', nargs=3, metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'LAYER'),
                help='find the sequence of visited clusters in a specified layer of the tree')
ap.add_argument('--read_clusterseq', nargs=1, metavar=('TAXI_ID'))
ap.add_argument('--read_log', nargs=1, metavar=('TAXI_ID'))
ap.add_argument('--delete', choices=['logs', 'sp', 'clusters'],
                help='delete a group within the HDF5 store from the choices listed')
args = ap.parse_args()


store = connect_to_store(*args.store)
if args.convert:
    initial_convert(
        store, *args.convert)  # initial convertion
if args.delete:
    delete_group(store, args.delete)
if args.spoints:
    calculate_sp(store, (int(args.spoints[0]), int(args.spoints[1])), float(
        args.spoints[2]), float(args.spoints[3]))  # stop point calculation
if args.cluster:
    cluster_sp(store, int(args.cluster[2]), float(args.cluster[3]), (int(
        args.cluster[0]), int(args.cluster[1])))  # stop point clustering
if args.clusterseq:
    create_cluster_sequences(store, (int(args.clusterseq[0]), int(
        args.clusterseq[1])), int(args.clusterseq[2]))  # cluster sequences
if args.read_clusterseq:
    read_clusterseq(store, int(*args.read_clusterseq))
if args.read_log:
    readstore_log(store, int(*args.read_log))


store.close()
