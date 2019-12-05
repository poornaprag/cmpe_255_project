import math
import random 
from collections import defaultdict
from pprint import pprint
import sys
#prevent future depreciation and warnning
import warnings 
warnings.filterwarnings(action='ignore')

#Basic libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("reddit_headlines_labels_2.csv")
df = df.append(df)
df = df.reset_index(drop=True)

# Getting rid of nuetrals
df = df[df.label!=0]
df['label'].value_counts()

#Transform headline into features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

headlines = df.headline.tolist()
print ("Headlines")
# print (headlines)
result_data = []

for item in headlines:
    result_data.append(item)

vect = CountVectorizer(binary=True)
X = vect.fit_transform(result_data)

vector_space = X.toarray().tolist()

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
nltk.download('vader_lexicon')
sia = SIA()
results = []

for i in range(len(headlines)):
    line = headlines[i]
    pol_score = sia.polarity_scores(line)
    summation = 0
    total = 0
    for num in vector_space[i]:
        total = total + 1
        summation = summation + num
    summation = summation/total
    results.append([summation, pol_score['pos'], pol_score['neg']])

data = np.array(results)
result_df = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1], 'Column3': data[:, 2]})

from sklearn.model_selection import train_test_split

X = result_df
y = df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print (len(X_train))
print (len(y_train))

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
print (X_train)
# sys.exit()
nb.fit(X_train, y_train)

nb.score(X_train, y_train)


y_pred = nb.predict(X_test)
# print (type(X_test))
y_pred

#f1 score calculated from harmonic mean with the help of confusion matrix
# it gives more analytic power than just accuracy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))

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