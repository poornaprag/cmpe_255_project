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



# @app.route("/about")
# def about():
#     return render_template('about.html')

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
    loaded_model = joblib.load('kn.pkl')
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
    print('what is here')
    print(what)
    return result


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['text']
        print(sentence)
        nb_result = naive_bayes(sentence)
        asdf()
        knn_result = knn(sentence)
        svm_result = svm(sentence)
        # prediction = [nb_result,knn_result,svm_result]
        prediction = []
        prediction.append(nb_result)
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
    df = pd.read_csv("comments_1.csv")
    df1 = pd.read_csv("comments_2.csv")
    df = df.append(df1)
    print(df.columns)
    df.columns=['x','y']
    # print(df)
    what = df['y'].value_counts()
    print(what)

    x=pd.DataFrame(what)
    x['x']=['-1','0','1']
    print(x)
    result = x.to_json(orient='records')
    # return result
    return render_template('visual.html', title='Visualize',result=result)


if __name__ == '__main__':
    app.run(debug=True)
