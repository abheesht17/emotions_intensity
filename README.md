## Predicting the Intensity of Emotions

## Problem Statement and Objective

- Supervised Learning Problem (Regression Task)
- Predict the degree of anger/joy in tweets from Twitter
- The score is a floating point number between 0 and 1

## Exploratory Data Analysis (EDA)

See ```EDA_Emotions_Intensity.ipynb```

## Methods
- We use three models: SVM, Decision Tree, MLP
- We try out four different embeddings using the models above:
	- Bag of Words (see ```approach_1```)
	- TF-IDF (see ```approach_1```)
	- BERT (see ```approach_2```)
	- BERT embeddings appended with statistical features calculated using hashtag:score and anger_words:score pairs from annotated datasets ([source](https://saifmohammad.com/WebPages/AccessResource.htm)) (see ```approach_3```)

## Preprocessing

- Bag of Words and TF-IDF Methods
	- Expand contractions
	- Remove URLs
	- Replace emoticons by <smile>, <lolface>,<sadface>,<neutralface>,<heart>, etc. Here, if represent emoticons as words using an annotated dictionary (as we did in BERT-Based Methods), it diminishes the performance.
	- lemmatize the text (gives significant gains in Pearson Coefficient) using nltk's WordNetLemmatizer and POS Tagging

- BERT-Based Methods
	- Expand contractions
	- Remove URLs
	- Replace emoticons with the corresponding words ([source for dictionary](https://github.com/NeelShah18/emot/blob/master/emot/emo_unicode.py))
	- StandardScaler() on the feature vectors did not give an improvement in the scores.


## Results

PEARSON (ANGER):

| METHOD       | SVM       | Decision Tree | MLP       | 
|--------------|-----------|---------------|-----------| 
| BAG OF WORDS | 0.583     | **0.44**      | 0.421     | 
| TF-IDF       | 0.532     | 0.436         | 0.539     | 
| BERT         | 0.6       | 0.307         | 0.58      | 
| BERT-STAT    | **0.613** | 0.168         | **0.641** | 


SPEARMAN (ANGER):

| METHOD       | SVM       | Decision Tree | MLP       | 
|--------------|-----------|---------------|-----------| 
| BAG OF WORDS | 0.572     | **0.41**      | 0.402     | 
| TF-IDF       | 0.519     | 0.409         | 0.511     | 
| BERT         | 0.6       | 0.3           | 0.587     | 
| BERT-STAT    | **0.615** | 0.159         | **0.642** | 


PEARSON (JOY):

| METHOD       | SVM       | Decision Tree | MLP       | 
|--------------|-----------|---------------|-----------| 
| BAG OF WORDS | 0.514     | 0.322         | 0.389     | 
| TF-IDF       | 0.532     | **0.324**     | 0.513     | 
| BERT         | **0.609** | 0.31          | 0.603     | 
| BERT-STAT    | 0.6       | 0.221         | **0.612** | 



SPEARMAN(JOY):

| METHOD       | SVM       | Decision Tree | MLP       | 
|--------------|-----------|---------------|-----------| 
| BAG OF WORDS | 0.512     | **0.331**     | 0.363     | 
| TF-IDF       | 0.536     | 0.316         | 0.506     | 
| BERT         | **0.618** | 0.319         | **0.608** | 
| BERT-STAT    | 0.6       | 0.217         | 0.604     | 






## Future Work

- Try out other transformers-based models (XLNet, RoBERTa, etc.) to compute the embeddings
- For the statistical methods (the one in which we compute the "angry words" and "hashtag" embeddings using annotated datasets), we can use PCA to reduce the dimensionality of those vectors
- Optimise models further by doing a Grid Search on more hyperparameters
- Find more statistical features we can use (such as the dataset with emoticon:score pairs)
- Find better ways to represent these statistical features
- OOP-ify the code :)