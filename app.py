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


def ValuePredictor(sample_test):
    # to_predict = np.array(to_predict_list).reshape(1, 12)
    # vocabulary_to_load = pickle.load(open(dictionary_filepath, 'r'))
    vect = joblib.load('vect.pkl')

    # loaded_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(ngram_size,
    # ngram_size), min_df=1, vocabulary=vocabulary_to_load)
    # loaded_vectorizer._validate_vocabulary()
    # loaded_vectorizer =  CountVectorizer(analyzer='word', binary=False, decode_error='strict',
    #     encoding='ISO-8859-1', input='content',
    #     lowercase=True, max_df=1.0, max_features=None, min_df=1,
    #     ngram_range=(2, 2), preprocessor=None, stop_words='english',
    #     strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
    #     tokenizer=tokenize, vocabulary=vocabulary_to_load)
    # print(loaded_vectorizer.)
    # print('loaded_vectorizer.get_feature_names(): {0}'.format(loaded_vectorizer.get_feature_names()))
    # loaded_model = pickle.load(open("bnb.pkl", "rb"))
    loaded_model = joblib.load('nb.pkl')
    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    print(result)
    # if result == 0:
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # to_predict_list = request.form.to_dict()
        sentence = request.form['text']
        print(sentence)
        # sentence = ''.join(sentence)
        # to_predict_list = list(to_predict_list.values())
        # to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(sentence)
        if result == 1:
            prediction = 'Bully'
        else:
            prediction = 'Non bully'
        return render_template("result.html", prediction=prediction)


@app.route("/data")
def data():
    # if username == '':
    #     username = 'realDonaldTrump'
    users_id = ['1425014252']
    usernames = []
    for id in users_id:
        usernames.append(api.get_user(id).screen_name)
    usernames.append('realDonaldTrump')
    return render_template('data.html', title='Data We Use', users=usernames)


@app.route('/data/<string:username>')
def data_user(username):
    def get_tweets(username):
        tweets = api.user_timeline(screen_name=username)
        return [{'tweet': t.text,
                 'created_at': t.created_at,
                 'username': username,
                 'headshot_url': t.user.profile_image_url}
                for t in tweets]

    return render_template('data.html', title='Data We Use', tweets=get_tweets(username))


@app.route("/visualize")
def vis():
    return render_template('visual.html', title='Visualize')


if __name__ == '__main__':
    app.run(debug=True)
