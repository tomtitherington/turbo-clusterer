#!/usr/bin/python3
import numpy as np
import pandas as pd

'''
dataframe format:
taxi id, date time, longitude, latitude
'''


def mean_coordinate(longs, lats, size):
    # int i = 0 ; i < size ; i+=1
    avg_longs = 0
    avg_lats = 0
    for i, j in zip(longs, lats):
        avg_longs += i
        avg_lats += j
    return (avg_longs / size, avg_lats / size)

''' time difference in minutes '''
def timespan(p_jt, p_it):
    (p_jt-p_it).astype('timedelta64[m]')
    return 0


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
    stop_points = {'taxi_id': [], 'date_time': [],
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
                    # split into arrival and leaving time?
                    stop_points['date_time'].append(df.iat[i, 1])
                    (long, lat) = mean_coordinate(
                        df.iloc[i:j, 2], df.iloc[i:j, 3], j - i + 1)
                    stop_points['longitude'].append(long)
                    stop_points['latitude'].append(lat)
                i = j
                break
            j += 1

    return pd.DataFrame(stop_points)


#print(distance((116.51172, 39.92123), (116.51135, 39.93883)))
df = pd.read_csv('release/taxi_log_2008_by_id/1.txt', header=None, names=[
                 "taxi_id", "date_time", "longitude", "latitude"])
timespan(df.iloc[0,1],df.iloc[1,1])
#print(mean_coordinate(df.iloc[0:3, 2], df.iloc[0:3, 3], 3 - 0 + 1))
