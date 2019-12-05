from flask import Flask, render_template, url_for, request
import tweepy as tw
import json
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import pickle
import joblib
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

@app.route("/")
@app.route("/how")
def how():
    return render_template('how.html', title='How It Works')


def tokenize(tweet):
    words = word_tokenize(tweet)

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in words]

    return words

def svm(sample_test):
    vect = joblib.load('vect1.pkl')
    loaded_model = joblib.load('sv.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    return result

def knn(sample_test):
    vect = joblib.load('vect1.pkl')
    loaded_model = joblib.load('lr.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    return result

def naive_bayes(sample_test):
    vect = joblib.load('vect1.pkl')
    loaded_model = joblib.load('nb.pkl')

    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    pdx = pd.DataFrame(result)
    what = pdx[0].value_counts().to_json(orient='records')

    return result

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['text']
        nb_result = naive_bayes(sentence)
        # asdf()
        knn_result = knn(sentence)
        svm_result = svm(sentence)

        prediction = []
        prediction.append(nb_result)
        prediction.append(knn_result)
        prediction.append(svm_result)
        
        return render_template("result.html", prediction=prediction)


@app.route("/data")
def data():
    return render_template('data.html', title='Data We Use')

@app.route("/visualize")
def vis():
    df = pd.read_csv("comments_1.csv")
    df1 = pd.read_csv("comments_2.csv")
    df = df.append(df1)
    df.columns=['x','y']

    data = df['y'].value_counts()

    x=pd.DataFrame(data)
    x['label']=['-1','0','1']
    
    sent = {'1' : 'Postive', '0' : 'Neutral', '-1': 'Negative'} 
        
    x.label = [sent[item] for item in x.label] 
    
    result = x.to_json(orient='records')
    # return result
    return render_template('visual.html', title='Visualize',result=result)


if __name__ == '__main__':
    app.run(debug=True)
