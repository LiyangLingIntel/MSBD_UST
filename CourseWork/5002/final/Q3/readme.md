# MSBD5002 Final Exam Question 3



For this question, my recognition is training data is not adequate could have two ways of explanation. 

* Missing data
* Actual dataset is not big enough 

#### Missing data

In the first part, missing data is very common problem for real_world data analysis. Like our course project of MSBD 5002, there are a lot of pieces of data are lost.

To solve this problem, we have a principle that real-world data is highly correlated, there is almost no data can exist independently. Thus, we could fill these missing data through their related data.

How to implements this idea?

* The simplest way, if only small part of data is missed, fill with the mean of other data in the same field.

* Analyze other data which have high correlation coefficients with target data, and built extra regression models to predict the missing data. 

  For example, in 5002 course project, to fill missing data, I use K-NN based regressor to generate proper data to fill in the empty position. The model input is the actual pisstion and actual time related features which I decided according to correlation analysis and basic knowledge about that area. In this way, we can fill missing data of one station from other stations in the same day and closed days.

#### Insuffitient size of dataset

In the second part, when the dataset itself is not big enough, the methods in part 1 cannot be applied. Some data augmentation techniques and skills could be adopted.

##### Image data

* Geometric transformation

  When dealing with images classification, it's common to meet situation with small size of training data. To solve this problem, a popular method is applying geometric transformation for some images, including flipping, cropping, rotating, scaling, and also prospect transforming to generate more images, but this method would lead to over-fitting.

* Adding noise

  Through adding Gaussian noise or sault and pepper noise to existing images to get new pictures, we can get some images with distored images which could enhance learning capacity to some extends.

* GANs

  Through training Generative adversarial networks on existing dataset, it could learn the characters, so that it could generate more similar images.

  For example, in my group project of course MSBD5012, we want to implement smile classification while only have 4000 fave images. We cope out mouths images and use GANs to extend the size of face image dataset.

##### Text data

​	Text data is not as many as image data which has many ways to augment.

* Tokenize sentence, shuffle tokens and rejoin to generate new sentences. Although the meaning of a text may be lost due to shuffling, it would not be an issue because feature extraction does consider order of words in a text.
* Synonym insertion on existing text data also could help to generate new data.

* There are also some ways like text encoding interpolations to generate text data like GANs in image augmentation.



##### References:

- https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9

- https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced
- https://www.sohu.com/a/198572539_697750

* https://towardsdatascience.com/data-augmentation-for-text-data-obtain-more-data-faster-525f7957acc9

* https://arxiv.org/pdf/1605.05396.pdf
* https://www.researchgate.net/post/Is_there_any_data_augmentation_technique_for_text_data_set