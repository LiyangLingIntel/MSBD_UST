##MSC-BDT5002  Answers to Assignment 2

## Income Prediction



### Feature Engineering

* **Missing Value Handling**

  I found there are many ' ?' value existing in nominal attributes, so this kind of missing value should be handled.

  Comparing the result of replacing with the mode of each feature, just replacing with empty string as a new value represent better performance.

  So I just replace them with empty value.

  For further works, some missing value on certain fields can be filled according to its most related features. For example, 'workclass' and 'race' are highly related to the 'education' and 'native-country' respectively. For some reasons, I have not try on this idea yet.

* **Nominal Features Handleling**

  Nominal feature cannot be fitted or examined directly, so I transfer them values to enumerate value and get their numerical value respectively.

  I have considered to use one-hot encode, but considering the model I use is adaboost whose base model is decision tree, one-hot encode has some overlap characteristics with decision tree, so I did not do that.
