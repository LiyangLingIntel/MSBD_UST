import math
import os
import time
import datetime
import pandas as pd
import numpy as np

from settings import data_root, output_folder, station_angle, station_dist


def load_data(file_names):

    if type(file_names) is not list:
        file_names = [file_names]
    df = None
    for file in file_names:
        temp_df = pd.read_csv(os.path.join(data_root, file))
        if df is None:
            df = temp_df
        else:
            df.append(temp_df)
    return df


def utc2local(utc_time: str):
    '''

    '''
    timeArray = time.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
    timestamp = time.mktime(timeArray)
    timestamp += 28800
    local_time = time.localtime(timestamp)
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    
    return local_time


def cal_dis(latitude1, longitude1,latitude2, longitude2):

    latitude1 = (math.pi/180)*latitude1
    latitude2 = (math.pi/180)*latitude2
    longitude1 = (math.pi/180)*longitude1
    longitude2= (math.pi/180)*longitude2

    R = 6378.1
    d =  math.acos(math.sin(latitude1)*math.sin(latitude2)+\
         math.cos(latitude1)*math.cos(latitude2)*math.cos(longitude2-longitude1))*R
    return d  # unit: km


def get_season(date: str):
    '''
    required date formate: 2018-11-11
    only get season in 2017 & 2018 by China's traditional Solar Terms
    '''
    date = date.split('-')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    
    if month in [12, 1]:
        return 'winter'
    elif month in [3, 4]:
        return 'spring'
    elif month in [6, 7]:
        return 'summer'
    elif month in [9, 10]:
        return 'autumn'
        
    if year == 2017:
        if month == 2:
            return 'winter' if day < 3 else 'spring'
        elif month == 5:
            return 'spring' if day < 5 else 'summer'
        elif month == 8:
            return 'summer' if day < 7 else 'autumn'
        elif month == 11:
            return 'autumn' if day < 7 else 'winter'
    elif year == 2018: 
        if month == 2:
            return 'winter' if day < 4 else 'spring'
        elif month == 5:
            return 'spring' if day < 5 else 'summer'
        elif month == 8:
            return 'summer' if day < 7 else 'autumn'
        elif month == 11:
            return 'autumn' if day < 7 else 'winter'
    return None


def get_solar_term(date: str, periodicity=False):
    '''
    date format: 2018-11-11
    mysterious formula: [Y*D+C]-L
    using idea of solar term to separate date to 24 parts, get label from 0 - 23
    using math.sin enable periodicity
    
    temperature rising:       2 --> 14    大暑最热 sin 最高点
    temperature dropping:     15 --> 1    大寒最冷 sin 最低点
    shift 7 units: 7->0, 24->7,
    '''
    solar_term = [
                    ["小寒", "大寒"],  # 1月        0  1  2
                    ["立春", "雨水"],  # 2月        2  3  4
                    ["惊蛰", "春分"],  # 3月        4  5  6
                    ["清明", "谷雨"],  # 4月        6  7  8
                    ["立夏", "小满"],  # 5月        8  9  10
                    ["芒种", "夏至"],  # 6月        10 11 12
                    ["小暑", "大暑"],  # 7月        12 13 14
                    ["立秋", "处暑"],  # 8月        14 15 16
                    ["白露", "秋分"],  # 9月        16 17 18
                    ["寒露", "霜降"],  # 10月       18 19 20
                    ["立冬", "小雪"],  # 11月       20 21 22
                    ["大雪", "冬至"]   # 12月       22 23 24
                ]

    D = 0.2422
    Cs = [5.4055, 20.12, 3.87, 18.73, 5.63, 20.646, 4.81, 20.1, 5.52, 21.04, 
          5.678, 21.37, 7.108, 22.83, 7.5, 23.13, 7.646, 23.042, 8.318, 23.438,
          7.438, 22.36, 7.18, 21.94]

    date = date.split('-')
    year = int(date[0]) % 100
    month = int(date[1])
    day = int(date[2])

    term_base = 2*month-1

    solar_day_1 = (year * D + Cs[term_base-1]) - ((year-1)/4)     # first solar term in current month
    solar_day_2 = (year * D + Cs[term_base]) - ((year-1)/4)     # second solar term in current month

    if day < solar_day_1:
        term_number = term_base - 1
    elif day < solar_day_2:
        term_number = term_base
    else:
        term_number = term_base + 1

    if term_number == 24: 
        term_number = 0 

    # shift
    term_shift = term_number + 7
    term_number = term_shift if term_shift <= 24 else term_shift-24
    
    if periodicity:
        size = 24
        period_param = math.sin((term_number/24)*math.pi)
        term_number = size * period_param

    return term_number


def get_time_label(time: str, periodicity=False):
    '''
    time: 12:00:00
    according to the hottest temperature at 14:00,
    02:00 - 14:00 0 --> 12
    14:00 - 02:00 11 --> 0
    '''
    hour = int(time.split(':')[0])
    #  2-->14  convert to 0-->12
    hour = hour-2 if hour-2>=0 else 2-hour
    if hour < 12:
        label = hour % 12
    else:
        label = 12 - hour % 12
    return label


def get_periodic_time_label(time: str):
    '''
    time: 12:00:00
    according to the hottest temperature at 14:00,
    02:00 - 14:00 0 --> 12
    14:00 - 02:00 11 --> 0
    '''
    hour = int(time.split(':')[0])
    #  2-->14  convert to 0-->12
    hour = hour-2 if hour-2>=0 else 24-(2-hour)
    size = 24
    period_param = math.sin((hour/24)*math.pi)
    label = size * period_param
    
    return label

# print(get_time_label('04:00:00'))

def cal_wind_projection(s1, s2, ws, wd):
    angle_name = f'{s1}-{s2}'
    angle = station_angle[angle_name]

    if wd > 360 or ws < 0.5:
        return 0
    else:
        included_angle = wd - angle
        projection_rate = math.cos(math.pi*included_angle/180)
        if projection_rate <= 0:
            return 0
        else:
            return ws*projection_rate


def get_max_dist(station, stations):
    stations = [i for i in stations if i != station]
    station_pairs = [sorted([station, i]) for i in stations]
    station_dists = [station_dist[f'{i[0]}-{i[1]}'] for i in station_pairs]
    return max(station_dists)


def construct_train_xy(df, target_station, station_list, only_x=False):
    if only_x:
        input_cols = ['station_id', 'test_id', 'temperature', 'pressure', 'humidity',
                      'periodic_solar_term', 'periodic_time', 'NearTraffic', 'Others', 'SubUrban', 'Urban', 'wind_speed']
    else:
        input_cols = ['station_id', 'temperature', 'pressure', 'humidity',
                      'periodic_solar_term', 'periodic_time', 'NearTraffic', 'Others', 'SubUrban', 'Urban', 'wind_speed']
    output_cols = ['PM2.5', 'PM10', 'O3']
    useful_cols = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_direction']

    qry = f'station_id=="{target_station}"'
    input_df = df.query(qry)[input_cols]
    input_df = input_df.reset_index().drop('index', axis=1)     # index other_df will make column cannot be combined
    
    max_dist = get_max_dist(target_station, station_list)
    for station in station_list:
        if station == target_station:
            continue
        short_stt = station.split('_')[0]
        station_pair = sorted([target_station, station])
        dist = station_dist[f'{station_pair[0]}-{station_pair[1]}']
        dist_rate = 1 - (dist/max_dist)
        
        qry = f'station_id=="{station}"'
        other_df = df.query(qry)[useful_cols]
        other_df = other_df.reset_index().drop('index', axis=1)

        input_df[f'{short_stt}_temperature'] = other_df['temperature'] * dist_rate
        input_df[f'{short_stt}_pressure'] = other_df['pressure'] * dist_rate
        input_df[f'{short_stt}_humidity'] = other_df['humidity'] * dist_rate
        
        wind_info = other_df[['wind_speed', 'wind_direction']].values.tolist()
        wind_info = [cal_wind_projection(station, target_station, x[0], x[1])*dist_rate for x in wind_info]
        input_df[f'{short_stt}_wind'] = np.array(wind_info) 
    
    if not only_x:
        return input_df.drop('station_id', axis=1), df.query(qry)[output_cols]
    else:
        return input_df

