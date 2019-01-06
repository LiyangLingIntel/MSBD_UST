

# MSBD5002 Final Exam Question 5

### Data

All my data is got from Twitter and Facebook, I get aroud 100 tweets, micro blogs and comments manually.

### Solutions

* **Processing text data**

  For common tweets and comments, there are many special punctuations and emojis. I removed them manually when get those texts, because the data size is not too large.

  Raw sentence is hard to extract good features, so firstly I use `TweetTokenizer` of  `nltk` to split each sentence or paragraphes into words.

  To reduce noise, I filtered word tokens by punctuation and `stopwords` of `nltk` and get leaner tokens. However, after later testing and comparation, model trained by sentence with stopwords could have better performance, so I removed the condition of stop words in the end.

* **Feature Engineering**

  Firstly, I used `word2vecter` method to transform my word tokens in text to word vectors in numberic array, and through trained word2vector model I can obtain the relations among words in corpus I collected. The total size of word2vec model vocabulary is 341.

  Secondly, I used `TfidfVectorizer` in `sk-learn`, which based on TF-iDF algorithm acquiring the importance of the words of my corpus. I could get the importance matrix of word tokens by its model training. the total size of important word tokens is 40.

  Finally, I multiplied the word vector of first part and importance matrix of second part to generate my final sentence feature matrix as training set. Because the vocabulary size is too small in `TfidfVectorizer` as a result of limited corpus, for those words could not be found I use one instead for simplification. 

* **Modeling**

  Because my manually picked texts do not have any labels, I chose to use unsupervised learning for this problems.

  Finally, K-Means was picked up as the final model. According to the requirements in exam paper, number of clusters was set as 2, and all sentences are tagged to 2 classes, label 1 is closer to positive sentiment while label 0 is closer to negetive sentiment.

   

#### Reference:

##### Data source:

https://twitter.com/search?q=gene%20editing&src=typd

https://twitter.com/search?q=genetic%20engineering&src=typd

https://twitter.com/search?q=transgene&src=typd

https://www.facebook.com/search/str/genetic+engineering/keywords_blended_posts?filters=eyJycF9hdXRob3IiOiJ7XCJuYW1lXCI6XCJtZXJnZWRfcHVibGljX3Bvc3RzXCIsXCJhcmdzXCI6XCJcIn0ifQ%3D%3D&epa=SEE_MORE

https://www.facebook.com/search/str/gene+editing/keywords_blended_posts?filters=eyJycF9hdXRob3IiOiJ7XCJuYW1lXCI6XCJtZXJnZWRfcHVibGljX3Bvc3RzXCIsXCJhcmdzXCI6XCJcIn0ifQ%3D%3D&epa=SEE_MORE

##### Idea reference:

https://www.jianshu.com/p/52ee8c5739b6

https://pythonspot.com/nltk-stop-words/

http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XAuN8BMzZ24

https://ahmedbesbes.com/sentiment-analysis-on-twitter-using-word2vec-and-keras.html

https://ahmedbesbes.com/how-to-mine-newsfeed-data-and-extract-interactive-insights-in-python.html