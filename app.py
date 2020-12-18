#TODO
# - confirm twitter auth is working as is [X]
# - figure out what the app.secret_key is about [X]
# - incorporate tweepy into new auth [X]
# - think about how are you going to deploy this cuz callback url is localhost [X]
#   - just create a prod app with ana actual url
# - get a twitter log in button functionality []
# - deploy

import io
import os
import base64
import pickle
import tweepy
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from dotenv import load_dotenv
from sentiment_model import SentimentModel
from flask import Flask, request, Response, render_template, redirect, url_for, session
from flask_dance.contrib.twitter import make_twitter_blueprint, twitter

# getting env variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
SECRET_KEY = os.getenv("SECRET_KEY")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")

# getting the model
model_r = keras.models.load_model('sentiment_model_lstm_v1/')

# setting up the app
app = Flask(__name__)
app.secret_key = SECRET_KEY 
blueprint = make_twitter_blueprint(api_key=API_KEY,
                                   api_secret=API_KEY_SECRET)
app.register_blueprint(blueprint, url_prefix="/login")
auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)


@app.route('/')
def home():
    if not twitter.authorized:
        return redirect(url_for('twitter.login'))

    resp = twitter.get('account/settings.json')
    print(resp.json()["screen_name"])
    return render_template('home.html')

@app.route('/single-tweet')
def tweet():
    return render_template('tweet.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    sm = SentimentModel(model_r)
    results = sm.predict_sentiments([message])
    print(f'predictions: {results}')
    return render_template('result.html',prediction = results[0])

@app.route('/scrape', methods=['POST', 'GET'])
def scrape():
    if request.method == "POST":
        auth.set_access_token(session.get('oauth_token'),
                              session.get('oauth_token_secret'))
        api = tweepy.API(auth)

        query = request.form['message']

        tweets = tweepy.Cursor(api.search,
                               q=query,
                               lang="en").items(500)
        tweets = [tweet.text for tweet in tweets]

        sm = SentimentModel(model_r)
        results = sm.predict_sentiments(tweets)


        labels, counts = np.unique(results, return_counts=True)
        plt.bar(labels, counts, width=.5, align='center')
        plt.gca().set_xticks(labels)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plot_url = 'data:image/png;base64,{}'.format(plot_url)

        return render_template('result_s.html',distribution = plot_url, query=query)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
