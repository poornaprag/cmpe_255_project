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
    loaded_model = joblib.load('kn.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    return result

def naive_bayes(sample_test):
    vect = joblib.load('vect1.pkl')
    loaded_model = joblib.load('nb.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['text']
        print(sentence)
        result = naive_bayes(sentence)
        knn_result = knn(sentence)
        svm_result = svm(sentence)
        # prediction = [nb_result,knn_result,svm_result]
        prediction = []
        prediction.append(result)
        prediction.append(knn_result)
        prediction.append(svm_result)
        print((prediction))
        # if result == 1:
            # prediction = 'Negative sentiment. Not safe to browse.'
        # else:
            # prediction = 'Safe to browse.'
        # prediction = ['abc','xyz','mav']
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


# @app.route('/data/<string:username>')
# def data_user(username):
#     def get_tweets(username):
#         tweets = api.user_timeline(screen_name=username)
#         return [{'tweet': t.text,
#                  'created_at': t.created_at,
#                  'username': username,
#                  'headshot_url': t.user.profile_image_url}
#                 for t in tweets]

#     return render_template('data.html', title='Data We Use', tweets=get_tweets(username))


@app.route("/visualize")
def vis():
    return render_template('visual.html', title='Visualize')


if __name__ == '__main__':
    app.run(debug=True)
