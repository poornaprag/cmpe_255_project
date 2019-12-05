import math
import random 
from collections import defaultdict
from pprint import pprint
import sys
import warnings 
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Transform headline into features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Reading in the data as data frame
df = pd.read_csv("reddit_headlines_labels_2.csv")
df = df.append(df)
df = df.reset_index(drop=True)

# Getting rid of neutrals
df = df[df.label!=0]
df['label'].value_counts()

# Getting headlines which we will use later to create vector count on
headlines = df.headline.tolist()
result_data = []

# Looping through each headline and adding as array as that is what the CountVecrotizer expects
for item in headlines:
    result_data.append(item)

# Getting Vector count of data
vect = CountVectorizer(binary=True)
X = vect.fit_transform(result_data)

# Converting to array for easy access later. We will use this to reduce the dimensions
vector_space = X.toarray().tolist()

# Used to get sentimented values (pos, neg and neutral) for each sentance
nltk.download('vader_lexicon')
sia = SIA()
results = []

# Creating new dataframe with reduced dimensions and add pos and neg
for i in range(len(headlines)):
    line = headlines[i]
    # Calculating pos and neg score
    pol_score = sia.polarity_scores(line)
    summation = 0
    total = 0
    # reducing dimensions by making into mean
    for num in vector_space[i]:
        total = total + 1
        summation = summation + num
    summation = summation/total
    results.append([summation, pol_score['pos'], pol_score['neg']])

# Creating dataframe of new data
data = np.array(results)
result_df = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1], 'Column3': data[:, 2]})

# Splitting and testing the data

# Splitting headers and labels for test and training data
X = result_df
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fitting data on naive bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb.score(X_train, y_train)

# Testing model
y_pred = nb.predict(X_test)
y_pred

#Testing Accuracy

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))


##############
# Method for testing a new record
##############
result_data = []

for item in headlines:
    result_data.append(item)
result_data.append("you are so bad!")

vect = CountVectorizer(binary=True)
X = vect.fit_transform(result_data)

vector_space = X.toarray().tolist()

pol_score = sia.polarity_scores("you are so bad!")
print (pol_score)
summation = 0
total = 0
for num in vector_space[-1]:
    total = total + 1
    summation = summation + num
summation = summation/total
test = [[summation, pol_score['pos'], pol_score['neg']]]

data = np.array(test)

final = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1], 'Column3': data[:, 2]})

y_pred = nb.predict(final)

y_pred

print (y_pred)