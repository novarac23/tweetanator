from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

test_setntance = "This food is amazing"

filename = 'ml_model_final.sav'
file = open(filename, 'rb')
loaded_model = joblib.load(file)

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
pr = PorterStemmer()

for i in range(0, 1):
    review = re.sub('[^a-zA-Z]', ' ', test_setntance)
    review = review.lower()
    review = review.split()
    review = [pr.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
vocabulary = joblib.load('vocabulary.pkl')
cv = CountVectorizer(max_features=2000, vocabulary=vocabulary)
X = cv.fit_transform(corpus).toarray()

result = loaded_model.predict(X)

if result[0] == 0:
    print("Negative")
elif result[0] == 4:
    print("Positive")
else:
    print("Not sure")

