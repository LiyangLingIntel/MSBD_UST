# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
from auxiliary import Deal_record, check_balance_warning, get_current_avg_price
from auxiliary import get_history_avg_price, generate_bar, generate_avg_bar
import pandas as pd
import numpy as np
import random
import os

from sklearn.externals import joblib
from keras.models import load_model
import statsmodels.api as sm
import pickle

# System Settings
ASSETS = ['BCH-USD', 'BTC-USD', 'ETH-USD', 'LTC-USD']
ORIGIN_FEATURES = ['close', 'high', 'low', 'open', 'volume']
ASSETS_NAME = ['BCH', 'BTC', 'ETH', 'LTC']
LONG, SHORT, HOLD = [1, 0, -1]
MODEL_PATH = './models/'

# User Settings
CONFIDENCE_SCOPE = 0.005
EXPECT_RETURN_RATE = 0.01
EXPECT_DEAL_RETURN = 0.004
DECISION_PERIOD = 60      # Number of minutes to generate next new bar
BAR_LENGTH = 120

CASH_BALANCE_LOWER_LIMIT = 20000    # cash balance problem!! if my cash balance rise back to 10,000 after sell the crypto, I still cannot make any deal.
CRYPTO_BALACE_LIMIT = 80000

PRIOR = 0.7     # the assumed chance that price will go up/down next week. get by analysis historcal data
PRIOR_WEIGHT = 0.2      # piror weight for each deal. 

# assumeed fluctunate volumn for each DECISION_PERIOD. get by analysis historical data. and should fit for each different currency
FLUCTUATE_VOLUMNS = [15, 100, 6, 1.3]
DEAL_UNITS = [6, 1, 8, 12]      # unit amount for each deal DECISION_PERIOD origin: [3, 1, 5, 10]

# Models Settings
rf_models = []
var_model = None
boost_models = []
xgb_models = []
lstm_models = []

# random forest
for asset in ASSETS_NAME:
    with open(f'models/model_{asset}.pickle', 'rb') as f:
        rf_models.append(pickle.load(f))
# var
var_model = sm.load('models/var_model_3')
# boost classification
for asset in ASSETS:
    boost_models.append(joblib.load(f'models/boost_clf_{asset}.pkl'))
# conbined xgboost
for model_name in ASSETS:
    m1 = joblib.load(os.path.join(MODEL_PATH, f'xg_{model_name.lower()}_0.joblib'))
    m2 = joblib.load(os.path.join(MODEL_PATH, f'xg_{model_name.lower()}_1.joblib'))
    m3 = joblib.load(os.path.join(MODEL_PATH, f'xg_{model_name.lower()}_2.joblib'))
    xgb_models.append([m1, m2, m3])
# lstm
lstm_models = []
for model_name in ASSETS:
    m = load_model(os.path.join(MODEL_PATH, f'lstm_{model_name}_model.h5'))
    lstm_models.append(m)


# decision_count=0
# failed_count = 0
# random.seed(123)
# is_reverse = False

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use

def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea

    # The idea of my strategy:
    # Logistic regression with label = rising/falling signal. 

    # Pattern for long signal:
    # for rising/falling prediction, long/short one unit of & according to the confidence to calculate the goal price, once it reachs sell the crypto.

    # TODO: embeded utility functions

    def get_income_rate(position, cryp_balance, current_price, transaction):
        return_balance = position * current_price
        transaction_cost = position * current_price * transaction
        if cryp_balance == 0:
            return_rate = 0
        elif cryp_balance > 0:
            return_rate = ((return_balance - cryp_balance) - transaction_cost) / cryp_balance
        else:
            return_rate = ((abs(cryp_balance) - return_balance) - transaction_cost) / cryp_balance
        return return_rate

    def get_confidence():
        pass

    # memory init
    if counter == 0:
        # data_save only saves data in the latest decision period (1 hour)
        memory.data_save_name = dict.fromkeys(ASSETS, pd.DataFrame(columns = ORIGIN_FEATURES))

        memory.deal_save = dict.fromkeys(ASSETS, [])
        memory.turning_price = dict.fromkeys(ASSETS, 0)     
        memory.invested_balance = dict.fromkeys(ASSETS, 0)      # total money spent on the cryptos
        memory.last_prediction = dict.fromkeys(['xgb', 'lstm'], 0) 

        memory.is_satisfied = False     # If True, stop to make deal
        memory.models_cof = [1 for i in range(4)]     # confidence for each model
        memory.use_model = [True for i in range(4)]   # use model or not (this is for ensembled model)

        memory.success_count = dict.fromkeys(ASSETS, 0)

        memory.data_save_index = []
        memory.data_cryp = []
        for i in range(4):
            #memory.data_save = np.zeros((DECISION_PERIOD, 5))#, dtype=np.float64)
            memory.data_save_index.append(pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume']))
            memory.data_cryp.append(pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume']))
        memory.hourly_rf = pd.DataFrame(columns = ['rate_BCH', 'dVolume_BCH', 'rate_BTC', 'dVolume_BTC',
                                                   'rate_ETH', 'dVolume_ETH', 'rate_LTC', 'dVolume_LTC'])
        memory.hourly_var = pd.DataFrame(columns=['BCH', 'BTC', 'ETH', 'LTC'])

    # data preprocess & record update
    position_new = position_current
    average_prices = np.mean(data[:, :4], axis=1)       # average price for all assets
    
    VAR = [0.5, 0.5, 0.5, 0.5]  # Prediction for var model
    RF = [0.5, 0.5, 0.5, 0.5]   # Prediction for Random Forest
    LR = [0.5, 0.5, 0.5, 0.5]   # Prediction for Logistic regression
    XGB = []
    LSTM = []


    if total_balance >= init_cash*(1+EXPECT_RETURN_RATE):
        memory.is_satisfied = True
    else:
        memory.is_satisfied = False

    ############# rf, var, boost classification #############

    if ((counter + 1) % DECISION_PERIOD == 0):
        bar = []
        for i in range(4):
            memory.data_save_index[i].loc[DECISION_PERIOD - 1] = data[i, ]
            line = generate_bar(memory.data_save_index[i])
            bar.append(line)
            memory.data_cryp[i] = pd.concat((memory.data_cryp[i], line), axis=0, sort=True)
            memory.data_cryp[i] = memory.data_cryp[i].reset_index(drop=True)
        price = []
        for t in range(DECISION_PERIOD):
            for i in range(4):
                time_price = memory.data_save_index[i].loc[t]
                time_price_mean = (time_price['close']+time_price['high']+time_price['low']+time_price['open'])/4
                price.append(time_price_mean)
        price = np.array(price)
        price = price.reshape(1, 240)

        LR = []
        for m in boost_models:
            LR.append(m.predict(price))

        open_ave = np.array([bar[0]['open_ave'].values[0], bar[1]['open_ave'].values[0], bar[2]['open_ave'].values[0],
                             bar[3]['open_ave'].values[0]])
        hourly_data = pd.DataFrame(data=[open_ave], columns=['BCH', 'BTC', 'ETH', 'LTC'])
        memory.hourly_var = pd.concat([memory.hourly_var, hourly_data], ignore_index=True)
        lag_order = var_model.k_ar

        if lag_order < len(memory.hourly_var.index):
            diff = np.log(memory.hourly_var).diff().dropna()
            X = diff.values[-lag_order:]
            pred = var_model.forecast(X, 2)
            baseline = 0
            for i in [0, 1, 2, 3]:
                if i == 1:
                    VAR[i] = -1
                    continue
                if (pred[0][i] > baseline and pred[1][i] > baseline): VAR[i] = 1
                if (pred[0][i] < (0 - baseline) and pred[1][i] < (0 - baseline)): VAR[i] = 0

    else:
        for i in range(4):
            memory.data_save_index[i].loc[(counter + 1) % DECISION_PERIOD - 1] = data[i,]

    df_cryp_sp = []
    for i in range(4):
        df_cryp = memory.data_cryp[i].copy()
        if len(df_cryp)>1:

            df_cryp['rate'] = (df_cryp['close'] -df_cryp['open']) / df_cryp['open']
            volume = pd.DataFrame(df_cryp['volume'][0:len(df_cryp)-1], columns=['volume'])
            df_cryp = df_cryp.drop(0).reset_index(drop = True)
            df_cryp['dVolume'] = abs((df_cryp['volume'] - volume['volume']) / volume['volume'])
            df_cryp = df_cryp[['rate', 'dVolume']]
            rateName = 'rate_' + ASSETS_NAME[i]
            dVolumeName = 'dVolume_' + ASSETS_NAME[i]
            df_cryp.columns = [rateName, dVolumeName]
            df_cryp_sp.append(df_cryp)
            memory.data_cryp[i] = memory.data_cryp[i].drop(0).reset_index(drop=True)
    if df_cryp_sp:
        cryp_data = pd.DataFrame()
        for cryp in df_cryp_sp:
            cryp_data = pd.concat((cryp_data, cryp), axis=1)
        memory.hourly_rf = pd.concat((memory.hourly_rf, cryp_data), axis=0).reset_index(drop=True)

    if len(memory.hourly_rf)>2:
        df_hourly = memory.hourly_rf.copy()
        for name in ASSETS_NAME:
            bins = [-1, -0.005, 0.01, 1]
            group_name = ['down', 'middle', 'up']
            predName = 'pred_' + name
            rateName = 'rate_' + name
            df_hourly[predName] = pd.cut(df_hourly[rateName], bins, labels=group_name)
            bins_volume = [0, 2.01, 100]
            group_volume_name = ['flat', 'sharp']
            dVolumeName = 'dVolume_' + name
            df_hourly[dVolumeName] = pd.cut(df_hourly[dVolumeName], bins_volume, labels=group_volume_name)
        df_hourly = df_hourly[
            ['pred_BCH', 'pred_BTC', 'pred_ETH', 'pred_LTC', 'dVolume_BCH', 'dVolume_BTC', 'dVolume_ETH',
             'dVolume_LTC']]
        df_cryp_t0 = pd.DataFrame(df_hourly.loc[0:len(memory.hourly_rf) - 3])
        df_cryp_t0.columns = ['BCH_t0', 'BTC_t0', 'ETH_t0', 'LTC_t0', 'dV_BCH_t0', 'dV_BTC_t0', 'dV_ETH_t0',
                              'dV_LTC_t0']
        df_cryp_t0 = df_cryp_t0.reset_index(drop=True)
        df_cryp_t1 = pd.DataFrame(df_hourly.loc[1:len(memory.hourly_rf) - 2])
        df_cryp_t1.columns = ['BCH_t1', 'BTC_t1', 'ETH_t1', 'LTC_t1', 'dV_BCH_t1', 'dV_BTC_t1', 'dV_ETH_t1',
                              'dV_LTC_t1']
        df_cryp_t1 = df_cryp_t1.reset_index(drop=True)
        df_hourly = df_hourly.drop([0,1]).reset_index(drop=True)
        df_hourly = pd.concat((df_hourly, df_cryp_t0), axis=1)
        df_hourly = pd.concat((df_hourly, df_cryp_t1), axis=1)
        X_train = df_hourly.drop(columns=['pred_BCH', 'pred_BTC', 'pred_ETH', 'pred_LTC'])

        X_train = pd.get_dummies(X_train)
        y = []
        for m in rf_models:
            y.append(m.predict(X_train))

        for i in range(4):
            if y[i] == 'up':
                RF[i] = 1
            elif y[i] == 'down':
                RF[i] = 0
            else:
                RF[i] = -1
        memory.hourly_rf = memory.hourly_rf.drop(0).reset_index(drop=True)

    # xgboost & LSTM
    # for each asset do the predict and make deal.
    for asset_index in range(4):

        # when achieving target income, stop making deals
        # TODO: OR when is_satisfied is true & next prediction is opposite result, clean position for more income
        if memory.is_satisfied:
            position_new[asset_index] = 0
            continue

        asset_name = ASSETS[asset_index]
        # fluctuate_volumn = FLUCTUATE_VOLUMNS[asset_index]
        deal_unit = DEAL_UNITS[asset_index]
        average_price = average_prices[asset_index]

        memory.data_save_name[asset_name].loc[counter] = data[asset_index,]
        
        # predict in DECISION_PERIOD
        if ((counter+1) % DECISION_PERIOD == 0  and counter>60):    

            # Risk Ananlysis
            if check_balance_warning(cash_balance, crypto_balance, total_balance, CASH_BALANCE_LOWER_LIMIT):
                continue

            # Model Evaluation
            if len(memory.deal_save[asset_name]) > 0:
                last_deal = memory.deal_save[asset_name][-1]
            else:
                last_deal = None
            # if decision_count > 150 and (failed_count*1.0) / decision_count > 0.7 and not is_reverse:
            if last_deal and int(memory.data_save_name[asset_name].iloc[-DECISION_PERIOD]['open']/memory.data_save_name[asset_name].iloc[-1]['close']) == last_deal.prediction:
                memory.success_count[asset_name] += 1
            else:
                memory.models_cof[asset_index] -= 0.01   # TODO better strategy needed
            if ((counter+1) % (DECISION_PERIOD*24*2) == 0):
                decision_count = int((counter+1) / DECISION_PERIOD)
                if memory.success_count[asset_name] / decision_count <= 0.5:
                    # memory.use_model[asset_index] = False
                    print(f'Not trust {asset_name} model now, but still use it')

            # Do prediction: use  model to predict or not
            if memory.use_model[asset_index]:
                '''
                What should do here:
                    load data for specified asset from memory.data_save[asset_name]
                    transform data format according to diff models
                    call model predict
                    ensemble result
                    give final prediction: 1: increasing, 0: decreasing, 2: hold the line (it's no matter without 2)
                '''
                # xgboost
                bar_x = generate_avg_bar(memory.data_save_name[asset_name], barlength=BAR_LENGTH)
                x_1 = []
                for j in range(4):
                    start = int(BAR_LENGTH*j/4)
                    end = int(BAR_LENGTH*(j+1)/4)
                    x_1.append(bar_x.values[start:end, 0].mean())

                y1 = xgb_models[asset_index][0].predict(x_1)
                y2 = xgb_models[asset_index][1].predict(x_1)
                y3 = xgb_models[asset_index][2].predict(x_1)
                predict_price = (y1+y2+y3)/3
                if predict_price > memory.last_prediction['xgb']:
                    XGB = 1
                elif predict_price < memory.last_prediction['xgb']:
                    XGB = 0
                else:
                    XGB = -1
                memory.last_prediction['xgb'] = predict_price

                # lstm
                bar_x = generate_avg_bar(memory.data_save_name[asset_name], barlength=BAR_LENGTH)
                bar_x = np.array([bar_x[['avg']].values[:,0]])
                x_2 = np.reshape(bar_x, (bar_x.shape[0], 1, bar_x.shape[1]))

                predict_price = lstm_models[asset_index].predict(x_2)
                if predict_price >= memory.last_prediction['lstm']:
                    LSTM = 1
                elif predict_price < memory.last_prediction['lstm']:
                    LSTM = 0
                else:
                    LSTM = -1
                memory.last_prediction['lstm'] = predict_price

            else:   # NOT sure is it a better way to replacce model prediction
                position_new[asset_index] = 0
                continue

            price_pred = [LR[asset_index], VAR[asset_index], RF[asset_index], XGB, LSTM]
            if price_pred.count(1) >= 2 and price_pred.count(0) <= 1:
                prediction = 1
            elif price_pred.count(0) >= 2 and price_pred.count(1) <= 1:
                prediction = 0
            else:
                prediction = -1

            # TODO: consider transaction fee and calculate income rate to refine deal unit
            if prediction == LONG:
                deal_type = 'long'
                if position_new[asset_index] > 0:
                    position_new[asset_index] += int(deal_unit * memory.models_cof[asset_index])
                    # Assume that new open = last close or avg
                    memory.invested_balance[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price       
                elif position_new[asset_index] == 0:
                    position_new[asset_index] += int(deal_unit * memory.models_cof[asset_index])
                    memory.turning_price[asset_name] = average_price
                    memory.invested_balance[asset_name] += int(deal_unit * memory.models_cof[asset_index]) * average_price
                else:
                    position_new[asset_index] = 0
                    memory.invested_balance[asset_name] = 0
            elif prediction == SHORT:
                deal_type = 'short'
                if position_new[asset_index] < 0:
                    position_new[asset_index] -= int(deal_unit * memory.models_cof[asset_index])
                    memory.invested_balance[asset_name] -= int(deal_unit * memory.models_cof[asset_index]) * average_price
                elif position_new[asset_index] == 0:
                    position_new[asset_index] -= int(deal_unit * memory.models_cof[asset_index])
                    memory.turning_price[asset_name] = average_price
                    memory.invested_balance[asset_name] -= int(deal_unit * memory.models_cof[asset_index]) * average_price       
                else:
                    position_new[asset_index] = 0
                    memory.invested_balance[asset_name] = 0
            elif prediction == HOLD: # HOLD
                deal_type = 'none'

            # record deal
            deal_price = memory.data_save_name[asset_name].iloc[-1]['close'] # unknown yet, for now just a assumed close price in this minute
            deal = Deal_record(amount=int(deal_unit * memory.models_cof[asset_index]))
            deal.prob_pred = memory.models_cof[asset_index]
            deal.asset_index = asset_index
            # deal.goal_price = deal_price + (1 if deal_type == 'long' else -1) * (PRIOR_WEIGHT * PRIOR + confidence) * fluctuate_volumn
            deal.deal_type = deal_type
            deal.prediction = prediction
            memory.deal_save[asset_name].append(deal)
        
        elif(counter != 0):   # not decision period & not the first minute
            # if current currency price can give double expect return, clean position
            return_rate = get_income_rate(position_new[asset_index], memory.invested_balance[asset_name], average_price, transaction)
            if return_rate >= EXPECT_DEAL_RETURN/2:
                position_new[asset_index] = 0
                memory.invested_balance[asset_name] = 0
            
    # End of strategy
    return position_new, memory
