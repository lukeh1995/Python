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

fake_news.head(10)
true_news.head(10)
print(fake_news.shape)
fake_news.describe()