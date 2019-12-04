from flask import Flask, render_template, url_for, request
import tweepy as tw
import json
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import pickle
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)


@app.route("/")
@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/how")
def how():
    return render_template('how.html', title='How It Works')




def ValuePredictor(sample_test):
    vect = joblib.load('vect.pkl')
    loaded_model = joblib.load('nb.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    print(result)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['text']
        print(sentence)
        result = ValuePredictor(sentence)
        if result == 1:
            prediction = 'Negative sentiment. Not very productive'
        else:
            prediction = 'No problem. '
        return render_template("result.html", prediction=prediction)


@app.route("/data")
def data():
    # if username == '':
    #     username = 'realDonaldTrump'
    # users_id = ['1425014252']
    # usernames = []
    # for id in users_id:
    #     usernames.append(api.get_user(id).screen_name)
    # usernames.append('realDonaldTrump')
    return render_template('data.html', title='Data We Use')




@app.route("/visualize")
def vis():
    return render_template('visual.html', title='Visualize')


if __name__ == '__main__':
    app.run(debug=True)
