#!/usr/bin/env python3
import numpy as np
import pandas as pd
import datetime


def mean_coordinate(longs, lats, size):
    avg_longs = 0
    avg_lats = 0
    for i, j in zip(longs, lats):
        avg_longs += i
        avg_lats += j
    return (avg_longs / size, avg_lats / size)


def timespan(p_it, p_jt):
    """Calculates the time difference between two entries in a log.

    Args:
        p_it: The datetime at point i.
        p_jt: The datetime at point j.
    Returns:
        The time difference in minutes.
    """
    return ((p_jt - p_it).seconds) / 60


def distance(p_i, p_j):
    """Calculates the distance in metres between two points.

    Uses the haversine formula to calculate the great-circle distance between two points
    of the form (longitude,latitude). φ is latitude, λ is longitude, R is earth’s radius
    (mean radius = 6,371km).

    Args:
        p_i: A tuple representing the first point.
        p_j: A tuple representing the second point.
    """
    R = 6371 * np.exp(3)  # meters
    phi_1 = p_i[1] * np.pi / 180  # radians
    phi_2 = p_j[1] * np.pi / 180
    delta_phi = (p_j[1] - p_i[1]) * np.pi / 180
    delta_lambda = (p_j[0] - p_i[0]) * np.pi / 180

    a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi_1) * \
        np.cos(phi_2) * np.sin(delta_lambda / 2) * np.sin(delta_lambda / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # in meters


def get_stops(df, df_size, dist_thresh, time_thresh):
    i = 0
    stop_points = {'taxi_id': [], 'arrival_dt': [], 'departure_dt': [],
                   'longitude': [], 'latitude': []}
    while i < df_size:
        #print("i: {}".format(i))
        j = i + 1
        print("i: {}".format(i))
        while j < df_size:
            #print("j: {}".format(j))
            dist = distance((df.iat[i, 2], df.iat[i, 3]),
                            (df.iat[j, 2], df.iat[j, 3]))
            # print(dist)
            if dist < dist_thresh:
                j += 1
                print("distance is less than tresh at j: {}".format(j))
            else:
                if j != (i + 1):
                    # the distance between i and j-1 is significant
                    j = j - 1
                    delta_t = timespan(df.iat[i, 1], df.iat[j, 1])
                    print("delta_t: {}".format(delta_t))
                    print("j: {}".format(j))
                    if delta_t > time_thresh:
                        print("significant")
                        # the time AND distance between i and j - 1 is significant
                        stop_points['taxi_id'].append(df.iat[i, 0])
                        stop_points['arrival_dt'].append(df.iat[i, 1])
                        stop_points['departure_dt'].append(df.iat[j, 1])
                        (long, lat) = mean_coordinate(
                            df.iloc[i:j + 1, 2], df.iloc[i:j + 1, 3], j - i + 1)
                        stop_points['longitude'].append(long)
                        stop_points['latitude'].append(lat)
                    i = j
                    break
                else:
                    # j == i + 1
                    i = j
                    break
        if not (j < df_size):
            break
        #i+=1
    return pd.DataFrame(stop_points)

# average of points within a threshold


def detect(df, df_size, dist_thresh, time_thresh):
    i = 0
    stop_points = {'taxi_id': [], 'arrival_dt': [], 'departure_dt': [],
                   'longitude': [], 'latitude': []}
    while i < df_size:
        j = i + 1
        while j < df_size:
            dist = distance((df.iat[i, 2], df.iat[i, 3]),
                            (df.iat[j, 2], df.iat[j, 3]))
            if dist > dist_thresh:
                delta_t = timespan(df.iat[i, 1], df.iat[j, 1])
                if delta_t > time_thresh:
                    stop_points['taxi_id'].append(df.iat[i, 0])
                    stop_points['arrival_dt'].append(df.iat[i, 1])
                    stop_points['departure_dt'].append(df.iat[j, 1])
                    (long, lat) = mean_coordinate(
                        df.iloc[i:j + 1, 2], df.iloc[i:j + 1, 3], j - i + 1)
                    stop_points['longitude'].append(long)
                    stop_points['latitude'].append(lat)
                i = j
                break
            j += 1
        break
        #i += 1

    return pd.DataFrame(stop_points)
