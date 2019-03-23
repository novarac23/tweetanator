import nltk
import numpy as np
import re
from flask import Flask,render_template,url_for,request
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
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

        return render_template('result.html',prediction = result[0])

if __name__ == '__main__':
    app.run(debug=True)
