#!/usr/bin/python3
import numpy as np
import pandas as pd
import datetime

'''
dataframe format:
taxi id, date time, longitude, latitude
'''


def mean_coordinate(longs, lats, size):
    avg_longs = 0
    avg_lats = 0
    for i, j in zip(longs, lats):
        avg_longs += i
        avg_lats += j
    return (avg_longs / size, avg_lats / size)


''' time difference in minutes '''


def timespan(p_it, p_jt):
    return ((p_jt - p_it).seconds) / 60


'''φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6,371km)
uses the ‘haversine’ formula to calculate the great-circle distance between two points
'''


def distance(p_i, p_j):
    R = 6371 * np.exp(3)  # meters
    phi_1 = p_i[1] * np.pi / 180  # radians
    phi_2 = p_j[1] * np.pi / 180
    delta_phi = (p_j[1] - p_i[1]) * np.pi / 180
    delta_lambda = (p_j[0] - p_i[0]) * np.pi / 180

    a = np.sin(delta_phi / 2) * np.sin(delta_phi / 2) + np.cos(phi_1) * \
        np.cos(phi_2) * np.sin(delta_lambda / 2) * np.sin(delta_lambda / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c  # in meters


''' df - the dataframe of a taxies complete trajectory '''


def stop_point_detection(df, df_size, dist_thresh, time_thresh):
    i = 0
    stop_points = {'taxi_id': [], 'arrival_dt': [], 'departure_dt': [],
                   'longitude': [], 'latitude': []}
    while i < df_size:
        j = i + 1
        while j < df_size:
            print(df.iat[j, 2], df.iat[j, 3])
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
        i += 1

    return pd.DataFrame(stop_points)


df = pd.read_csv('release/taxi_log_2008_by_id/1.txt', header=None, names=[
                 "taxi_id", "date_time", "longitude", "latitude"])
df['date_time'] = pd.to_datetime(df['date_time'])
# distnace threshold 50 meters, time threshold 3 minutes
print(stop_point_detection(df, len(df.index), 50, 3))
