# MSBD5002 Final Exam Question 1



### Inbalance problem

number of inlier pics: 4801
number of outlier pics: 413
number of testing pics: 1300

There are two ways to solve this problems:

* using GAN to generate more outliers
* resample inliers to a to a reasonable amount
* or combine the above two methods

First problem is, if use GAN, the amount of outlier pictures needed to generate exceed the amount of origin data too much, it may import much noise to the training set if GAN model is not good enough.

So I Choose the second plan. To garrantee the amount of training set and testing set keeping on the same level, I chose inlier : outlier = 2.5 : 1 to resample inlier set and get 1613 pictures as my final train data source. So that my training set is not so inbalance and training size is larger than test set.

Although, maybe combining GAN and resampling is a good way, for this time-limited exam, simplify the problem is also a good way.

### Feature Extraction

As the experience in Assignment 3, to extract features of pictures, I adopted pretrained model from *imagenet* on Keras.

Among those models, I chose MobileNet  as my final picture processing model.

There has several steps:

* Resize pictures to 224*224
* Expand resized pictures to one dimension and process resized picture with input processor
* Use resnet model to extract 2048 features for each picture
* Label outlier pictures with 1 and inliers with 0

### Modeling

* Basically I chose XGBoost Classifier as my classification model.
* Because models based on tree algorithms are not easy to be affected by data inbalance, the parameter of booster I decide is 'gbtree'.
* After comparing the number of estimators from 500 to 2000 and maximum depth from 5 to 1, I finally pick up 600 estimators and 3 level depth as input hyper parameters, and get validation accuracy as 94.117%

For the last step, construct output dataframe with picture name and the relative label. output to output.csv