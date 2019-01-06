## MSBD 5002 Data Mining - Assignment 3



### Overview

1. Use pre-trained model extract features from images
2. Evaluate clustering performance with different K values
3. Using best performed feature-extracting model and K to cluster images



#### General Steps

* To improve efficiency, randomly pick 20 percent from 5010 images as samples.

* Import models pre-trained with imagenet in keras, like `VGG16`, `VGG19`, `ResNet50`, `DenseNt169`, and `MobileNet` to process images and to extract features.
* Using KMeans to do clustering basted on extracted features in last step. Intuitively choose K value between 9 and 15.
* Using silhouette analysis to determin the best suitable K.
* Apply the final feature extracting model and K onto complete image set, and get cluster labels.

#### Basic Result

From efficiency and accuracy aspects, I chose ResNet50 as my final feature extracting model, and chose 12 as my final K.

The  average silhouette_score is : 0.047678523, very low value, maybe because poorly feature engeering.