#!/usr/bin/python3
import pandas as pd
import tables

import stop_point_detection as spd

#TODO: convert each csv into single hdf5 with the correct types in the dataframes
def initial_convert(arg):
    pass


def connect_to_store(filename):
    return HDFStore(filename)


def read_log(filename):
    df = pd.read_csv(filename, header=None, names=[
        "taxi_id", "date_time", "longitude", "latitude"])
    df['date_time'] = pd.to_datetime(df['date_time'])
    return df


def store_sp_from_csv(store, n):
    for i in range(1, n):
        log = read_log("release/taxi_log_2008_by_id/{}".format(i)
        # distance threshold = 50 meters - time threshold = 3 minutes
        sp = spd.detect(log,len(log.index),50,3)
        store.append("sp/{}".format(i),sp)
    store.close()
