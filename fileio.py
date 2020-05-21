#!/usr/bin/python3
import pandas as pd
import tables
import os
import os.path
import argparse

import stop_point_detection as spd
import point_clustering as pc


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
    for i in range(1, n + 1):
        log = store.get('logs/t{}'.format(i))
        sp = spd.detect(log, len(log.index), delta_d, delta_t)
        store.append("sp/t{}".format(i), sp, format='table', index=False)
    store.close()


def cluster_sp(store, type, taxi_id=None, plot=None):
    if taxi_id is None:
        return "Not yet implemented"
    df = store.get('sp/t{}'.format(taxi_id))
    pc.build_tree(df,10,0.001)
    store.close()
    return



# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--convert", nargs=2, required=True,
#                help="Convert the files specified first arg (directory) into a single HDF5 with \
#                name specified in second arg")
# print(vars(ap.parse_args()))

filename = "taxi_store.h5"
# initial_convert("taxi_store.h5", "release/taxi_log_2008_by_id/")
# calculate_sp(connect_to_store(filename),10, 50, 3)
cluster_sp(connect_to_store(filename),'birch',1,'y')
