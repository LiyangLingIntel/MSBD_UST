

# MSBD5002 Final Exam Question 8



### Solutions

#### Data Analysis

There are three data file in the resource, users data, movie data and rating information.

Our target is using the combination of these three datasets and make  classification through user id and movie id.

There several jobs have been down.

#### Feature Engineering

##### User based

* I generally used one hot encoding to the separete the `gender` column.
* Relabelize `Age` column, transform sparse integer to continuous numeric values
* Cut bins for `zip-code`. For zip code can show the main state or area information, I labelized zip code by its first three numbers, and used 678 continuous number for `zip-code`.

##### Movie based

* Extract publish time for each movie from its title and drop the origin title
* Using adjusted one hot encoding to transform `Genres` to 0|1 tags in multi types. In this way, movie catogories can be showed clearly.

##### Others

As some new features have been added to users and movies dataset, the next is dealing with rating data set and aggregating those data and features together.

* Convert origin timestamps to localtime label, specificly, to local year. So that, some comparation could be made between publish time.
* All the features in user data and movie data are merged to rating date by the `UserID` and `MovieID`respectively. 
* What's more, a lot of aggregations have been made. 
  * Through aggregation, the average of rating score and made by each `MovieID`
  * Also, according to `UserID` & `MovidID`, the average rating score for each gender is made to represent the preference for different gender
  * There are still lots of ways to enrich this problem, Like aggragate from occupations, from age and etc. Even we could use the aggregated min, max and avg from different conditions to make this problems to a approximate linear regression problems.

Because of time limitation, there are still have a lot things could be done to improve accuracy, but no time to cope with them.

### Modeling

For this question, I chose XGBoost as my final model.

The predicting result is 1-5 numeric values. The simplest way is to do 5-classification. However, the 5 levels of label is increasing related. For example, the difference between 1 and 5 and between 4 and 5 is apparently different.

So, I used regression method to figure out this problem. That is, basically, predict score by regression model and then convert float score to numeric integer values.

In the end, the model validation result is not very good. The accuracy is around 40% while `rmse` is around 85%.

