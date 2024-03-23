# setup
# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from math import floor

# loading dataset
# url = 'https://drive.google.com/file/d/1O6PAzQd808rWNxkyL3ToO3qpaBlvQ2zC/view?usp=share_link'
# path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
# df = pd.read_csv(path, index_col=0)
df = pd.read_csv('gpt-tweet-sentiment.csv', index_col=0)

# analyzing dataset
# Example EDA: Sentiment distribution
# sns.countplot(data=df, x="labels")
# plt.title('Distribution of Sentiment Classes')
# plt.show()

# preprocessing
# extract the tweets input feature and the labels output feature (good, bad, neutral)
tweets = df['tweets']
tweets = np.array(tweets)
labels = df['labels']
labels = np.array(labels)


# function to clean tweets of
# clean tweets of hashtags, mentions, links, emojis, literal '\n' and any other characters that are not in the alphabet
def clean_tweet(tweet):
    # Convert the tweet to lowercase
    cleaned_tweet = tweet.lower()

    # Expand contractions
    cleaned_tweet = contractions.fix(cleaned_tweet)

    # Remove literal '\n' string
    cleaned_tweet = cleaned_tweet.replace(r'\n', '')

    # Remove links
    cleaned_tweet = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_tweet)

    # Remove mentions
    cleaned_tweet = re.sub(r'@\w+', '', cleaned_tweet)

    # Remove hashtags
    cleaned_tweet = re.sub(r'#\w+', '', cleaned_tweet)

    # Remove emojis
    cleaned_tweet = cleaned_tweet.encode('ascii', 'ignore').decode('ascii')

    # Remove punctuations
    cleaned_tweet = re.sub(r'[^\w\s]', '', cleaned_tweet)

    # remove non alphabet characters
    cleaned_tweet = re.sub(r'[^a-zA-Z\s]', '', cleaned_tweet)

    # Remove numbers
    cleaned_tweet = re.sub(r'\d+', '', cleaned_tweet)

    # Remove extra spaces
    cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet).strip()

    return cleaned_tweet


# function to get random sample of a numpy array of input features and numpy array of output features
def get_random_sample(input_features, output_features, sample_size):
    # Shuffle the indices of the dataset
    num_samples = input_features.shape[0]
    shuffled_indices = np.random.permutation(num_samples)

    # select a random sample of indices
    random_sample_indices = shuffled_indices[:sample_size]

    # Extract the corresponding samples from both input and output arrays
    input_sample = input_features[random_sample_indices]
    output_sample = output_features[random_sample_indices]
    return input_sample, output_sample


# going through all tweets and cleaning them, and removing stop words
for x in range(0, len(tweets)):
    tweets[x] = clean_tweet(tweets[x])
    tweets[x] = remove_stopwords(tweets[x])

# changing output features (labels) to numbers: good=1, neutral=2, bad =3
Y = []
for label in labels:
    if label == "good":
        Y.append(1)
    elif label == "neutral":
        Y.append(2)
    else:
        Y.append(3)
# converting Y into numpy array
Y = np.array(Y)

# getting random sample of data because it is too big to train and test on my machine (getting 10% of data)
test_sample_size = floor((len(tweets) * 0.1))
random_input_sample, random_output_sample = get_random_sample(tweets, Y, test_sample_size)
print(f"random input sample: {random_input_sample} \n")
print(f"random output sample: {random_output_sample} \n")

# tokenization of the tweets, remove words/features that only show up once, and vectorization of the words
vectorizer = CountVectorizer(min_df=2)
X = vectorizer.fit_transform(random_input_sample)
Y = random_output_sample
# printing feature info
print("-" * 100)
print(f" number of input features: {X.shape[1]}")
print("-" * 1000)
print(f" rows: {X.shape[0]}")
print(f" columns: {X.shape[1]}")
print("length of output features", len(Y))

# splitting data into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# modeling (SVM)
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(x_train, y_train)
y_prediction_training = clf.predict(x_train)
y_prediction_test = clf.predict(x_test)

print("\n------------------ Evaluation metrics of SVM model from sklearn ------------------")
# the accuracy score for the training and test data
print(f"Accuracy score for the training data: {accuracy_score(y_train, y_prediction_training)}\n")
print(f"Accuracy score for the test data: {accuracy_score(y_test, y_prediction_test)}\n")

# the recall scores for the training and test data
print(f"Recall scores for the training data: {recall_score(y_train, y_prediction_training, average=None)} \n")
print(f"Recall scores for the test data: {recall_score(y_test, y_prediction_test, average=None)} \n")

# the f1 score for the training and test data
print(f"f1 scores for the training data: {f1_score(y_train, y_prediction_training, average=None)}\n")
print(f"f1 scores for the test data: {f1_score(y_test, y_prediction_test, average=None)} \n")

# the precision score for the training and test data
print(f"Precision scores for the training data: {precision_score(y_train, y_prediction_training, average=None)}\n")
print(f"Precision scores for the test data: {precision_score(y_test, y_prediction_test, average=None)}")
print("----------------------------------------------------------------------------------\n")

# Modeling (NN)

# summary

# conclusion
