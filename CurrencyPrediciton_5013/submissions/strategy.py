# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
from sklearn.externals import joblib
from auxiliary import Deal_record, check_balance_warning, get_current_avg_price, get_history_avg_price, generate_bar
import pandas as pd
import numpy as np
import random

use_model = False
model = joblib.load('./model.pkl')
period = 60 * 4  # Number of minutes to generate next new bar
decision_count=0
failed_count = 0

cash_balance_lower_limit = 20000 #20000 # cash balance problem!! if my cash balance rise back to 10,000 after sell the crypto, I still cannot make any deal.
crypto_balance_limit = 80000

piror = 0.7 # the assumed chance that price will go up/down next week. get by analysis historcal data
piror_weight = 0.2 # piror weight for each deal. 
# fluctuate_volumn_for_each_crypto = [50, 300, 20, 2] # assumeed fluctunate volumn for each period. get by analysis historical data. and should fit for each different currency
fluctuate_volumn_for_each_crypto = [15, 100, 6, 1.3]
deal_unit_for_each_crypto = [3, 1, 5, 10] # unit amount for each deal period
random.seed(123)

is_reverse = False

# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use

# [[close, high, low, open], --'BCH-USD'
#  [close, high, low, open], --'BTC-USD'
#  [close, high, low, open], --'ETH-USD'
#  [close, high, low, open]] --'LTC-USD'
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

    global deal_unit   
    global piror
    global piror_weight
    global period
    global failed_count
    global decision_count
    global is_reverse
    global use_model

    # Get position of last minute
    position_new = position_current

    average_price_for_all_assets = np.mean(data[:, :4], axis=1)
    if counter == 0:
        memory.data_save = dict()
        memory.deal_save = list()

    # for each asset do the predict and make deal.
    for asset_index in range(4):

        # if counter > 60 * 24 * 5 and total_balance > 105000:
        # if total_balance > 105000:
        #     position_new[asset_index] = 0
        #     continue

        average_price = average_price_for_all_assets[asset_index]
        fluctuate_volumn = fluctuate_volumn_for_each_crypto[asset_index]
        deal_unit = deal_unit_for_each_crypto[asset_index]
        # Generate OHLC data for every 30 minutes
        if (counter == 0):
            # memory.data_save = np.zeros((bar_length, 5))#, dtype=np.float64)
            memory.data_save[asset_index] = pd.DataFrame(columns = ['close', 'high', 'low', 'open', 'volume'])
            # to support define outlier deals.
            memory.open_price = average_price
            memory.period_highest = dict()
            memory.period_lowest = dict()

        # check if any deal can sell in usual time.
        if len(memory.deal_save) > 0: 
            for deal in memory.deal_save:
                if deal.asset_index != asset_index: continue
                if deal.is_dirty_deal and not deal.has_dropped:
                    deal.drop_price = average_price
                    deal.has_dropped = True
                if deal.has_selled or deal.has_dropped: continue
                if not deal.has_brought:
                    # update record
                    deal.deal_time = counter
                    deal.price = average_price 
                    deal.goal_price += deal.price
                    deal.has_brought = True
                    deal.has_selled = False
                else:
                    # process deal, see if could sell.
                    if deal.deal_type == 'long' and average_price >= deal.goal_price:
                        # sell long
                        if not check_balance_warning(cash_balance, crypto_balance, total_balance, cash_balance_lower_limit, deal):
                            position_new[asset_index] -= 1 * deal_unit
                            deal.sell_time = counter
                            deal.has_selled = True
                    if deal.deal_type == 'short' and average_price <= deal.goal_price:
                        # sell long
                        if not check_balance_warning(cash_balance, crypto_balance, total_balance, cash_balance_lower_limit, deal):
                            position_new[asset_index] += 1 * deal_unit
                            deal.sell_time = counter
                            deal.has_selled = True
        
        # predict in period
        if ((counter + 1) % period == 0):
            if check_balance_warning(cash_balance, crypto_balance, total_balance, cash_balance_lower_limit):
                continue
            memory.data_save[asset_index].loc[period - 1] = data[asset_index,]


            # wether use a model to predict.
            if use_model:
                # use the model to predict
                bar = generate_bar(memory.data_save[asset_index]) # pandas dataframe
                bar_X = bar[['open', 'close']]
        
                prob_pred = model.predict_proba(bar_X)[:,1][0]
                # prob_pred = -1 # random.random()
                is_up = prob_pred > 0.51
                is_down = prob_pred < 0.49
            else:
                # don't use model. judge trend based on the average price of last period & current price.
                hist_avg_price = get_history_avg_price(memory.data_save[asset_index].drop('volume', axis=1), period)
                curr_avg_price = get_current_avg_price(data[asset_index,][:4])

                prob_pred = 1
                is_up = hist_avg_price < curr_avg_price
                is_down = hist_avg_price > curr_avg_price

            if decision_count > 150 and (failed_count*1.0) / decision_count > 0.7 and not is_reverse:
                is_reverse = True

            if is_reverse:
                is_up = not is_up
                is_down = not is_down

            make_deal = is_up or is_down
            confidence = prob_pred
            deal_type = 'none'

            if is_up:
                deal_type = 'long'
                position_new[asset_index] += 1 * deal_unit
            elif is_down:
                deal_type = 'short'
                position_new[asset_index] -= 1 * deal_unit

            if make_deal:
                decision_count += 1
                # store deal
                deal_price = 0 # unknown yet, for now just a assumed price in this minute
                # TODO should also consider transaction fee.
                goal_price = deal_price + (1 if deal_type == 'long' else -1) * (piror_weight * piror + confidence) * fluctuate_volumn
                # deal = Deal_record(base_time='2018-08-01 00:00:00')
                deal = Deal_record(base_time='2018-10-07 00:00:00')
                deal.prob_pred = prob_pred
                deal.asset_index = asset_index
                deal.goal_price = goal_price
                deal.deal_type = deal_type
                deal.has_selled = False
                deal.has_brought = False
                memory.deal_save.append(deal)
        
            if len(memory.deal_save) > 0: 
                for deal in memory.deal_save:
                    if not deal.has_brought: continue
                    if deal.has_selled: continue
                    if deal.is_hold_till_end: continue
                    if deal.is_dirty_deal: continue
                    if (deal.deal_time) % period == 0:
                        failed_count += 1
                        # deal.is_dirty_deal = True
                        a_index = deal.asset_index
                        # open_price_of_asset = memory.open_price[a_index]
                        if piror > 0: # piror is the price will go up in final.
                            # TODO deal with outliers.
                            # if (deal.price < open_price_of_asset or (deal.price + pow((1 + piror), 2) * fluctuate_volumn) > open_price_of_index):
                            if deal.deal_type == 'long':
                                # hold till end and keep finding chance to sell
                                deal.is_hold_till_end = True
                            elif deal.deal_type == 'short':
                                # sell directly to stop loss
                                position_new[a_index] += 1 * deal_unit
                                deal.is_dirty_deal = True
                                pass
                    
                        if piror < 0: # piror is the price will go down in final.
                            # TODO deal with outliers.
                            # if (deal.price > open_price_of_asset or (deal.price + pow((1 + piror), 2) * fluctuate_volumn > open_price_of_index):
                            if deal.deal_type == 'long':
                                # sell directly to stop loss
                                position_new[a_index] -= 1 * deal_unit
                                deal.is_dirty_deal = True
                                pass
                            elif deal.deal_type == 'short':
                                # hold till end and keep finding chance to sell
                                deal.is_hold_till_end = True
        
        else:
            memory.data_save[asset_index].loc[(counter + 1) % period - 1] = data[asset_index,]#####

    # End of strategy
    return position_new, memory
