import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.externals import joblib

dataset = pd.read_csv("smol_tweets.csv", encoding="UTF-8") # 0 - negative, 4 - positive

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
pr = PorterStemmer()

for i in range(0, 200000):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    review = [pr.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

feature_list = cv.get_feature_names()
joblib.dump(feature_list, 'vocabulary.pkl')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

filename = 'ml_model_final.sav'
joblib.dump(classifier, filename)