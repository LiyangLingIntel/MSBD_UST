# MSBD5002 Final Exam Question 6



### Solutions

For this question, there several jobs should be done. Video stream data processing, video feature extraction  and classification model training. Below are my detailed steps.

##### Data processing

For dealing with video stream data directly, I do not have any experience. Then inspired by ideas from the internet, I simplify this problem from video classification to image classification.

* **Image extraction**

  Based on `opencv-python`, I cut video at timepoint of 1/4, 2/4 and 3/4 of the video length respectively and get three pictures for each video. The reason of chosing timepoints like this is that through my observation and common recognition, at the beginning and very end of a video, the lightness, color of it  would not be very normal, even there will be some logos or slogans which could add in lots of noise. If the timepoints are too close, it's also not good to show the average content in that video. Thus I set timepoint in this way.

* **Feature engineering**

  Based on the experience of former image processing, I also chose to use pre-trained model to extract features. For this problem, I adopted keras `MobileNet` which is trained by *imagenet* as feature extractor because of its high working speed and excellent extranting performance.

  So, I separately extract features from three images of each video. After dimision reduction, I got 1024 features for each picture, combine them together and form 3072 dimensions matrix as the training input.

* **Modeling**

  For this part, based on the experience on other related projects, I would also chose XGBoost classifier as my final classification models.

  To reduce the effects of small inbalance problem among the videos in different tags in training data, I chose *gbtree* as the booster of XGBoost, 600 estimators and maximum depth as 3.

  Finally get validation accuracy 85.956% and considerable good classification result.


#### References

https://zhuanlan.zhihu.com/p/48419604