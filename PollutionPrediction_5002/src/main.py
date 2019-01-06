import os
import pandas as pd
from utils import *
from settings import output_folder, model_folder
from settings import station_dist, station_angle, station_locale, station_list
# from test_data import test_df

from sklearn.externals import joblib

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder

test_df = pd.read_csv(os.path.join(output_folder, 'test.zip'))


if __name__ == "__main__":
    
    output_df = pd.DataFrame()
    for target_station in station_list:
        station_name = target_station.split('_')[0]
        input_x = construct_train_xy(test_df, target_station, station_list, only_x=True)
        output = input_x[['station_id', 'test_id']]
        input_x = input_x.drop(['station_id', 'test_id'], axis=1)
        m_o3_1 = joblib.load(os.path.join(model_folder, f'{station_name}_o3_2.joblib'))
        m_o3_2 = joblib.load(os.path.join(model_folder, f'{station_name}_o3_3.joblib'))
        m_pm10_1 = joblib.load(os.path.join(model_folder, f'{station_name}_pm10_1.joblib'))
        m_pm25_1 = joblib.load(os.path.join(model_folder, f'{station_name}_pm25_1.joblib'))
        m_pm25_2 = joblib.load(os.path.join(model_folder, f'{station_name}_pm25_2.joblib'))
        m_pm25_3 = joblib.load(os.path.join(model_folder, f'{station_name}_pm25_3.joblib'))
        
        output['O3'] = (m_o3_1.predict(input_x)+m_o3_2.predict(input_x))/2
        output['PM2.5'] = (m_pm25_1.predict(input_x)+m_pm25_2.predict(input_x)+m_pm25_3.predict(input_x))/3
        output['PM10'] = m_pm10_1.predict(input_x)
        
        output_df = output_df.append(output)

    output_df = output_df[['test_id', 'PM2.5', 'PM10', 'O3']]
    output_df.to_csv(os.path.join(output_folder, 'submission.csv'), index = False)
    print('Output submission to outpus folder.')
