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


def initial_convert(filename, data_path):
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
    store = connect_to_store(filename)
    for name in os.listdir(data_path):
        fullpath = os.path.join(data_path, name)
        if os.path.isfile(fullpath):
            store.append("logs/t{}".format(os.path.splitext(name)[0]), read_log(fullpath),
                         data_columns=['date_time'], format='table')
    store.close()


def store_sp_from_csv(store, n):
    for i in range(1, n):
        log = read_log("release/taxi_log_2008_by_id/{}.txt".format(i))
        # distance threshold = 50 meters - time threshold = 3 minutes
        sp = spd.detect(log, len(log.index), 50, 3)
        store.append("sp/{}".format(i), sp)
    store.close()


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
    for i in range(n[0], n[1]):
        try:
            log = store.get('logs/t{}'.format(i))
        except:
            print("File logs/t{} could not be opened".format(i))
            continue
        sp = spd.detect(log, len(log.index), delta_d, delta_t)
        store.append("sp/t{}".format(i), sp, format='table', index=False)
    store.close()


def cluster_sp(store, order, threshold, r):
    tree = cft.CFTree(order, threshold)
    # read each taxis stop points in range r
    for i in range(r[0], r[1]):
        try:
            df = store.get('sp/t{}'.format(i))
        except:
            print("File sp/t{} could not be opened".format(i))
            continue
        lnglats = df[['longitude', 'latitude']]
        for index, row in lnglats.iterrows():
            tree.insert_point(row.values)
    tree.save_tree(store)
    store.close()

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
            #print("index: {}".format(index))
            c_index = index
            c_distance = dist
        index += 1
    # print('c index: {}'.format(c_index))
    return clusters.iat[c_index, 0]


def create_cluster_sequence(store, taxi, layer):
    try:
        clusters = store.get('clusters/l{}'.format(layer))
        taxi_sp = store.get('sp/t{}'.format(taxi))
    except:
        print("Clusters at layer {} could not be opened".format(layer))
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
    store.close()
    print(clusters)


def delete_clusters(store):
    store.remove('clusters/')
    store.close()
    # print(store.groups())


def delete_sps(store):
    store.remove('sp/')
    store.close()


def delete_group(store, group):
    store.remove(group)
    store.close()


ap = argparse.ArgumentParser()
# Default action to complete the whole process or just create the file?
# MOBAL
ap.add_argument('store', metavar='STORE', nargs=1,
                help='the name of the HDF5 store file')
ap.add_argument('--delete', choices=['logs', 'sp', 'clusters'],
                help='delete a group within the HDF5 store from the choices listed')
ap.add_argument('--convert', metavar='INDIR', nargs=1,
                help='the directory path of the csv file(s) to be converted and placed in the stores')
ap.add_argument('--spoints', metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'DISTANCE_THRESH', 'TIME_THRESH'), nargs=4,
                help='calculate stop points within the specified range, with time and distance threshold')
ap.add_argument('--cluster', nargs=4, metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'ORDER', 'THRESH'),
                help='cluster the stop points of taxis in a specified range with branching factor and threshold')
ap.add_argument('--clusterseq', nargs=3, metavar=('LEFT_BOUND', 'RIGHT_BOUND', 'LAYER'),
                help='find the sequence of visited clusters in a specified layer of the tree')
ap.add_argument('--read_clusterseq', nargs=1, metavar=('TAXI_ID'))
args = ap.parse_args()

if args.delete:
    delete_group(connect_to_store(*args.store), args.delete)
if args.convert:
    initial_convert(
        *args.store, *args.convert)  # initial convertion
if args.spoints:
    calculate_sp(connect_to_store(
        *args.store), (args.spoints[0], args.spoints[1]), args.spoints[2], args.spoints[3])  # stop point calculation
if args.cluster:
    cluster_sp(connect_to_store(
        args.store[0]), int(args.cluster[2]), float(args.cluster[3]), (int(args.cluster[0]), int(args.cluster[1])))  # stop point clustering
if args.clusterseq:
    create_cluster_sequences(connect_to_store(
        store), (args.clusterseq[0], args.clusterseq[1]), args.clusterseq[2])  # cluster sequences
if args.read_clusterseq:
    read_clusterseq(connect_to_store(*args.store), *args.read_clusterseq)


filename = "taxi_store.h5"

""" initial_convert """
# initial_convert("taxi_store.h5", "release/taxi_log_2008_by_id/")

""" stop point calculation """
# calculate_sp(connect_to_store(filename),1000, 50, 3)
# delete_sps(connect_to_store(filename))

""" clustering """
# cluster_sp(connect_to_store(filename),50, 0.5, (1,1000))
# read_clusters(connect_to_store(filename),0)
# delete_clusters(connect_to_store(filename))


""" cluster sequences """
# create_cluster_sequence(connect_to_store(filename), 300, 1)
# print(get_sp(connect_to_store(filename),300))
