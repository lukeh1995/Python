#Import required libraries
import os
import numpy as np
import pandas as pd
import re
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
porter_stemmer = PorterStemmer()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Load fake and real datasets
fake_news = pd.read_csv(r'C:\Users\60234651\Downloads\archive\Fake.csv')
true_news = pd.read_csv(r'C:\Users\60234651\Downloads\archive\True.csv')

#Initial data exploration
fake_news.head(10)
true_news.head(10)
print(fake_news.shape)
fake_news.describe()
fake_news['subject'].value_counts()

#Plot diffent subjects for both
fake_news['subject'].value_counts().plot(kind='bar')
true_news['subject'].value_counts().plot(kind='bar')

#Encode fake and real news as 1, 0
fake_news['type'] = 1
true_news['type'] = 0

#Combine data into one dataframe
news_data = [fake_news, true_news]
news = pd.concat(news_data)

#Merge title with text of news article as this is also of interest in building our model
news['text'] = news['title'] + ' ' + news['text']
#Drop title and date columns as not of use to analysis
news = news.drop(['title', 'date'], axis = 1)

#First we will use stemming to reduce the text to it's root, creating a function to do so
def stem_sentences(sentence):
    tokens = sentence.split()
    stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

news["Stemmed Text"] = news['text'].apply(stem_sentences)

news.head(10)

#Declare X and Y variables - X being the stemmed text, and Y being the type of text (real or fake)
X = news["Stemmed Text"].values
Y = news['type'].values

#Next we will use TF-IDF for feature extraction
vect = TfidfVectorizer()

#Apply vectorizer to X
vect.fit(X)
X = vect.transform(X)

#Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=100)

#Building Model
clf = svm.SVC(kernel='linear') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))