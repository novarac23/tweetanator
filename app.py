import nltk
import numpy as np
import re
import os
from flask import Flask,render_template,url_for,request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from functools import wraps
from flask import request, Response

app = Flask(__name__)

@app.route('/')
def home():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
    else:
        return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
    else:
        if request.method == 'POST':
            message = request.form['message']

            filename = 'ml_model_final.sav'
            file = open(filename, 'rb')
            loaded_model = joblib.load(file)

            corpus = []
            pr = PorterStemmer()

            for i in range(0, 1):
                review = re.sub('[^a-zA-Z]', ' ', message)
                review = review.lower()
                review = review.split()
                review = [pr.stem(word) for word in review if not word in set(stopwords.words('english'))]
                review = ' '.join(review)
                corpus.append(review)

            vocabulary = joblib.load('vocabulary.pkl')
            cv = CountVectorizer(max_features=2000, vocabulary=vocabulary)
            X = cv.fit_transform(corpus).toarray()

            result = loaded_model.predict(X)

            if result[0] == 0:
                result = "Negative"
            elif result[0] == 4:
                result = "Positive"
            else:
                result = "I don't know"

            return render_template('result.html',prediction = result)


def check_auth(username, password):
    username = os.environ.get('username')
    password = os.environ.get('password')
    return username == username and password == password

def authenticate():
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})


if __name__ == '__main__':
    app.run(debug=True)
