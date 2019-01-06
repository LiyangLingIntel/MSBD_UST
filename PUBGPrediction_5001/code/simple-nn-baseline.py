import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('./pubg/train.csv')
df_test = pd.read_csv('./pubg/test.csv')

"""
it is a team game, scores within the same group is same, so let's get the feature of each group
"""
df_train_size = df_train.groupby(['matchId','groupId']).size().reset_index(name='group_size')
df_test_size = df_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')

df_train_mean = df_train.groupby(['matchId','groupId']).mean().reset_index()
df_test_mean = df_test.groupby(['matchId','groupId']).mean().reset_index()

df_train_max = df_train.groupby(['matchId','groupId']).max().reset_index()
df_test_max = df_test.groupby(['matchId','groupId']).max().reset_index()

df_train_min = df_train.groupby(['matchId','groupId']).min().reset_index()
df_test_min = df_test.groupby(['matchId','groupId']).min().reset_index()

"""
although you are a good game player, 
but if other players of other groups in the same match is better than you, you will still get little score
so let's add the feature of each match
"""
df_train_match_mean = df_train.groupby(['matchId']).mean().reset_index()
df_test_match_mean = df_test.groupby(['matchId']).mean().reset_index()

df_train = pd.merge(df_train, df_train_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_mean, suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del df_train_mean
del df_test_mean

df_train = pd.merge(df_train, df_train_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_max, suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del df_train_max
del df_test_max

df_train = pd.merge(df_train, df_train_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_min, suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del df_train_min
del df_test_min

df_train = pd.merge(df_train, df_train_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
df_test = pd.merge(df_test, df_test_match_mean, suffixes=["", "_match_mean"], how='left', on=['matchId'])
del df_train_match_mean
del df_test_match_mean

df_train = pd.merge(df_train, df_train_size, how='left', on=['matchId', 'groupId'])
df_test = pd.merge(df_test, df_test_size, how='left', on=['matchId', 'groupId'])
del df_train_size
del df_test_size

target = 'winPlacePerc'
train_columns = list(df_test.columns)

""" remove some columns """
train_columns.remove("Id")
train_columns.remove("matchId")
train_columns.remove("groupId")
train_columns.remove("Id_mean")
train_columns.remove("Id_max")
train_columns.remove("Id_min")
train_columns.remove("Id_match_mean")

"""
in this game, team skill level is more important than personal skill level 
maybe you are a good player, but if your teammates is bad, you will still lose
so let's remove the features of each player, just select the features of group and match
"""
train_columns_new = []
for name in train_columns:
    if '_' in name:
        train_columns_new.append(name)
train_columns = train_columns_new    
print(train_columns)

# train_columns = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 
#                 'heals', 'killPlace', 'killPoints', 'kills', 'killStreaks', 
#                 'longestKill', 'maxPlace', 'numGroups', 'revives','rideDistance', 
#                 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 
#                 'weaponsAcquired', 'winPoints']

x_train = df_train[train_columns]
x_test = df_test[train_columns]
y = df_train[target]

del df_train

x_train = np.array(x_train, dtype=np.float64)
x_test = np.array(x_test, dtype=np.float64)
y = np.array(y, dtype=np.float64)

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
# scaler = preprocessing.QuantileTransformer().fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x_train = x_train*2 - 1
# x_test = x_test*2 - 1
y = y*2 - 1

print("x_train", x_train.shape, x_train.min(), x_train.max())
print("x_test", x_test.shape, x_test.min(), x_test.max())
print("y", y.shape, y.min(), y.max())

x_test = np.clip(x_test, a_min=-1, a_max=1)
print("x_test", x_test.shape, x_test.min(), x_test.max())

from ultimate.mlp import MLP 
mlp = MLP(layer_size=[x_train.shape[1], 28, 28, 28, 1], regularization=1, output_shrink=0.1, output_range=[-1,1], loss_type="hardmse")

"""
train 15 epoches, batch_size=1, SGD
"""
mlp.train(x_train, y, verbose=2, iteration_log=20000, rate_init=0.08, rate_decay=0.8, epoch_train=15, epoch_decay=1)
pred = mlp.predict(x_test)
pred = pred.reshape(-1)

pred = (pred + 1) / 2
# df_test['winPlacePerc'] = np.clip(pred, a_min=0, a_max=1)

df_test['winPlacePercPred'] = np.clip(pred, a_min=0, a_max=1)
# df_test['winPlacePercPred'] = pred
aux = df_test.groupby(['matchId','groupId'])['winPlacePercPred'].agg('mean').groupby('matchId').rank(pct=True).reset_index()
aux.columns = ['matchId','groupId','winPlacePerc']
df_test = df_test.merge(aux, how='left', on=['matchId','groupId'])
    
submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)