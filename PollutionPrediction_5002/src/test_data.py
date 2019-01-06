import os
import pandas as pd
from utils import *
from settings import data_root, output_folder, report_folder
from settings import station_dist, station_angle, station_locale, station_list

from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

'''
This part is converted from jupyter notebook manually,
to some extend, its structure is more ipython, not python
'''

# Construct test dataset
gwt = pd.read_csv(os.path.join(output_folder, 'grid_weather_test.zip'))
owt = pd.read_csv(os.path.join(output_folder, 'ob_weather_test.zip'))

useful_cols = ['longitude', 'latitude', 'local_time', 'temperature', 'pressure', 'humidity',
               'wind_direction', 'wind_speed']

df_indicator = gwt[useful_cols]
df_indicator = df_indicator.append(owt[useful_cols])

localtimes = gwt['local_time'].unique()

# generate most basic feature columns
test_df = pd.DataFrame()
for s in station_list:
    d = {}
    d['station_id'] = [s for i in range(48)]
    d['test_id'] = [f'{s}#{i}' for i in range(48)]
    d['local_time'] = localtimes
    locale_info = station_locale[s]
    d['longitude'] = [locale_info[0][0] for i in range(48)]
    d['latitude'] = [locale_info[0][1] for i in range(48)]
    d['area'] = [locale_info[1] for i in range(48)]
    df = pd.DataFrame(d)
    test_df = test_df.append(df)

test_df['date'] = test_df['local_time'].map(lambda x: x.split()[0])
test_df['time'] = test_df['local_time'].map(lambda x: x.split()[1])
test_df['periodic_solar_term'] = test_df['date'].map(lambda x: get_solar_term(x, periodicity=True))
test_df['periodic_time'] = test_df['time'].map(lambda x: get_periodic_time_label(x))
test_df = test_df.drop(['area'], axis=1).join(pd.get_dummies(test_df['area']))

# add more data from past days for knn regression
gw_df = pd.read_csv(os.path.join(output_folder, 'grid_weather_processed.zip'))
ow_df = pd.read_csv(os.path.join(output_folder, 'ob_weather_processded.zip'))
df_indicator_old = gw_df[useful_cols]
df_indicator_old = df_indicator.append(ow_df[useful_cols])

indicators = gw_df[useful_cols].append(ow_df[useful_cols]).query('local_time > "2018-04-28 07:00:00"')
indicators = indicators.append(df_indicator)

# delete useless data to free space
del df_indicator_old
del gw_df
del ow_df

# add datetime_label for knn to tag time distance
datetime_labeler = LabelEncoder()
indicators['datetime_label'] = datetime_labeler.fit_transform(indicators['local_time'])
test_df['datetime_label'] = datetime_labeler.transform(test_df['local_time'])

# use knn regressor to transfer each weather condition to target stations
idct_list = ['temperature','pressure','humidity','wind_direction','wind_speed']
for indic in idct_list:
    neibours = 7 if indic in ['temperature','pressure','humidity'] else 5
    weight = 'distance' # 'uniform', 'distance'
    train_x = indicators[['longitude', 'latitude', 'datetime_label']]
    train_y = indicators[indic]
    knn = KNeighborsRegressor(n_neighbors=neibours, weights=weight)
    knn.fit(train_x, train_y)
    test_df[indic] = knn.predict(test_df[['longitude', 'latitude', 'datetime_label']])

# get final uncleaned test set
test_df = test_df.drop_duplicates(subset=['test_id'], keep='first')
test_df.to_csv(os.path.join(output_folder, 'test.zip'), compression='zip')