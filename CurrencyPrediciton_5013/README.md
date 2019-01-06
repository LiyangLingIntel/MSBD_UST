# msbd5013_project
msbd5013 group project repository

Give real world virtual currencies like Bit-Coin every week, we need to train our own model and apply specified strategy to make prediction and earn some money (pretend to earn).

## Folder Usage

* data

Contains both format1 & format2 data from dropbox.

* analysis

Historical data analysis should be doing in this folder.

* model

Model building should be done in this folder.

* submissions

All submitted history project result archieve.

* others

Other support and helpful files or pages should be placed here.

## File Usage

* backTest.py

Enterance for porgramming running.

(Data file directory have already be setted as './data/')

* backTest.ipynb

backTest in jupyter-notebook format.

(Also contains some debug code for strategy.)

* strategy.py

Strategy code.

* auxiliary.py

Support functions for stategy.


## Helpful Links.
* about time series analysis : https://www.cnblogs.com/foley/p/5582358.html

* some source code about strategy and model: https://www.joinquant.com

* others/Analysis of Financial Time Series.pdf

* others/Time series prediction using dynamic Bayesian network.pdf

* [LSTM-Neural-Network-for-Time-Series-Prediction](https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction)

# Meeting Summary

## 2018/10/04 会议结论

1. Strategy

    1.1 完善Cash_balance监控，要分清long，short时候对position的影响及cash的影响，保证不被平仓。

    1.2 急涨急跌的应急处理。细节完善
   
    1.3 联系TA关于清仓后，cash高于10000后，应该依然可以做交易的问题


2. Model

    2.1 基于Bayesian Network的初步建模。
   
    > 输入(t-1)时刻四个货币的价格, 输出为(t))时刻四种货币价格或者走势的预测。


3. Statistics & Analysis

    3.1 多个货币之间的走势联系和影响，分析及利用。
    
    > 合适的预测粒度（多少小时、或者天），以支持模型训练及优化
    
    > 走势“领头”数字货币，支持策略方面实时计算决策。

    3.2 交易量变化分析 - volume & price 走势关联分析 

    3.3 急涨急跌后的走势分析，辅助决策。


4. Arragement
  
    Strategy - ZHANG Xichen

    S & A - GAO Han

    Model - WANG Shen, LING Liyang (coding support)

5. Deadline

    2018/10/18 (Week.3)


## 2018/10/11 会议结论

1. Lask week rank: No.1

2. Model

    2.1 Linear Regression.
    
    > model/model_linear_regression.ipynb
    
    > Y = (W.T)*X, where X is the data at time (T-n) of all 4 assests, Y is the predict assest price at time T.

    2.2 Logistic Regression.
    
    > modify based on linear regression model    

3. Strategy

    based on model, if the predict is trustable. trade.

4. Arragement

    1. Linear Regression parameter extend to (T-1), (T-2) ... (T-n) - ZHANG Xichen

    2. Logistic Regression Modification. - GAO Han

    3. Linear Regression parameter extend to volume. - LING Liyang

    4. Textbook method attempt. - WANG Shen

5. DDL

    2018/11/01 (Week.5)
