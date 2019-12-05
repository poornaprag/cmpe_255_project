from flask import Flask, render_template, url_for, request
# import tweepy as tw
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


def tokenize(sub):
    words = word_tokenize(sub)

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

def nb2(sample_test):
    vect = joblib.load('vect2.pkl')
    loaded_model = joblib.load('nb2.pkl')

    sample_test_dtm = vect.transform([sample_test])
    result = loaded_model.predict(sample_test_dtm)
    pdx = pd.DataFrame(result)
    what = pdx[0].value_counts().to_json(orient='records')

    return result

def get_sample_data():
    d1 = pd.read_csv("comments_1.csv")
    d2 = pd.read_csv("comments_2.csv")
    d3 = d1.append(d2)
    df=pd.DataFrame(d3)
    
    from sklearn.model_selection import train_test_split

    X = df.headline
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    from sklearn.feature_extraction.text import CountVectorizer

    vect1 = CountVectorizer(ngram_range=(1,1))

    vect2 = CountVectorizer(ngram_range=(2,2))

    X_train_vect = vect1.fit_transform(X_train)

    # making equal samples of -1 and +1 by oversampling with the help of SMOTE
    # Handling class imbalance
    from imblearn.over_sampling import SMOTE

    sm = SMOTE()

    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    import numpy as np
    unique, counts = np.unique(y_train_res, return_counts=True)
    
    result = list(zip(unique, counts))
    negative = None
    positive = None
   
    if result[0][0] == -1:
        negative = result[0][1]
        positive = result[1][1]
    else:
        negative = result[1][1]
        positive = result[0][1]
    final_res = [{"y":int(positive),"label":"Postive"},{"y":int(negative),"label":"Negative"}]

    return final_res

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        sentence = request.form['text']
        nb_result = naive_bayes(sentence)
        knn_result = knn(sentence)
        svm_result = svm(sentence)
        nb2_result = nb2(sentence)

        prediction = []
        prediction.append(nb_result)
        prediction.append(knn_result)
        prediction.append(svm_result)
        prediction.append(nb2_result)
        
        return render_template("result.html", prediction=prediction, sentence=sentence)


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
    x['label']=['0','-1']
    
    sent = {'0' : 'Postive', '-1': 'Negative'} 
        
    x.label = [sent[item] for item in x.label] 
    
    result = x.to_json(orient='records')
    # return result
    import json 
    sample_data = json.dumps(get_sample_data())
    print (result)
    print (sample_data)
    return render_template('visual.html', title='Visualize',result=result, sample_data=sample_data)


if __name__ == '__main__':
    app.run(debug=True)
