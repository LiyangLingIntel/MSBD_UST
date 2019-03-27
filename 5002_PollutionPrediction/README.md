# AirQuality Prediction

Group ID: 50

Liyang Ling

2053 7456

### code

In `src/` there are all my codes. 

Contains two parts: 

First is rerunable code which could import all models I trained to generate `submission.csv`

Second is my analysis and processing codes, they cannot be rerun because of the lack of source data and middle data files.

#### Rerunable files:

* mian.py -- the entry of whole program which could generate prediction result
* Utils.py -- contains all of my reusable functions and functional functions
* Settings.py -- static variables and pre calculated data which could reduce program run time

#### Data processing files:

* EDA_MissingData_handler.ipynb -- handles missing data
* EDA_Feature_Engineering.ipynb -- does feature engineering
* FE_model_selection.ipynb -- chooses and evaluates different models
* model_training.ipynb -- trains final models
* make_prediction.ipynb -- Jupyter notebook version of main.py but can't be reran

* test_data.py -- processes and saves test data 

### File Tree Structure

* `src/` : all my codes
* `models/` : 210 models for prediction
* `outputs` : partial of my outputs: one for test data & three for reducing calculation complexity
* report.pdf



PS: Because sth wrong with these files, I compress all models and middle  output files in one file `models&output.zip`, to rerun my code, please decompress this file and move two subfiles to root folder.

PSS: Because the dataset files are too large to email, so I only packaged necessaty files, if you want to run all of my code, please email me to resend those middle data or rerun processing files in above order.