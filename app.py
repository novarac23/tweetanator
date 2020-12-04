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
from flask import Flask, request, Response, render_template

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_SECRET = os.getenv("API_KEY_SECRET")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")


auth = tweepy.OAuthHandler(API_KEY, API_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

app = Flask(__name__)
model_r = keras.models.load_model('sentiment_model_lstm_v1/')

@app.route('/')
def home():
    return render_template('home.html')

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
        query = request.form['message']

        tweets = tweepy.Cursor(api.search,q=query,lang="en").items(500)
        tweets = [tweet.text for tweet in tweets] 

        sm = SentimentModel(model_r)
        results = sm.predict_sentiments(tweets)

        plt.hist(results)
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plot_url = 'data:image/png;base64,{}'.format(plot_url)

        return render_template('result_s.html',distribution = plot_url, query=query)

    return render_template('tweet_scraper.html')

if __name__ == '__main__':
    app.run(debug=True)
