from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
import numpy as np


class Deal_record(object):

    def __init__(self, amount=1, base_time = '2018-09-01 00:00:00', time_format='%Y-%m-%d %H:%M:%S'):
        self.base_time = base_time
        self.time_format = time_format
        self.deal_type = 'none' # long or short
        self.amount = amount
        self.asset_index: int = -1
        self.deal_time: int = 0
        self.sell_time: int = 0
        self.price = 0 # deal price
        self.prob_pred = 0 # predicted probality
        self.goal_price = 0    # goal price to sell
        self.has_selled = False
        self.has_brought = False
        self.is_hold_till_end = False
        self.is_dirty_deal = False # wrong deal. cannot sell
        self.has_dropped = False
        self.drop_price = 0
    

    def __str__(self):
        base_time = dt.strptime(self.base_time, self.time_format)
        time_in = base_time + timedelta(seconds=60*self.deal_time) if self.has_brought else 0
        time_out = (base_time + timedelta(seconds=60*self.sell_time)) if self.sell_time != 0 else 0
        # "time_in,time_out,deal_type,prob_pred,asset_index,in_price,goal_price,has_brought,has_selled,is_dirty_deal,is_hold_till_end\n" 
        return f"{time_in},{time_out},{self.deal_type},{self.prob_pred},{self.asset_index},{self.price},{self.goal_price},{self.has_brought},{self.has_selled},{self.is_dirty_deal},{self.has_dropped},{self.drop_price},{self.is_hold_till_end}"



def check_balance_warning(current_balance:int, crypto_balance:int, total_balance:int, limit:int, deal:Deal_record=None) -> bool:
    if deal == None:
        if current_balance < limit:
            return True
        else:
            return False
    else:
        return False
        #if total_balance - crypto_balance - deal.goal_price * deal.amount < limit:
        #    return True
        #else:
        #    return False


def get_current_avg_price(data):
    average_price = np.mean(data)
    return average_price


def get_history_avg_price(data, period):
    average_price = np.mean(data[-1*period:], axis=1)
    return average_price.mean()


# def generate_bar(data):
#     # pandas dataframe
#     open_price = data['open'][0]
#     close = data['close'][len(data) - 1]
#     high = data['high'].max()
#     low = data['low'].min()
#     volume_ave = data['volume'].mean()
#     bar = pd.DataFrame(data = [[open_price, high, low, close, volume_ave]], columns = ['open', 'high', 'low', 'close', 'volume_ave'])
#     return bar 

def generate_bar(data, barlength=120):
    ## Data is a pandas dataframe
    # open_price = data['open'][0]
    # close = data['close'][len(data) - 1]
    # high = data['high'].max()
    # low = data['low'].min()
    # volume_ave = data['volume'].mean()
    # price_avg = data[['close', 'high', 'low', 'open']].mean(axis=1)
    df_bar = data.iloc[-barlength:, :].copy()
    df_bar['avg'] = df_bar[['close', 'high', 'low', 'open']].mean(axis=1)
    # OHLC = pd.DataFrame(data = [[close, high, low, open_price, price_avg, volume_ave]], columns = ['close', 'high', 'low', 'open', 'avg', 'volume'])
    return df_bar

