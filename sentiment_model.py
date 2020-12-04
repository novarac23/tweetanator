import pickle
import numpy as np

from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

class SentimentModel:
    def __init__(self, model, tokenizer_path='tokenizer.pickle', max_length=21):
        self.model = model

        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle) 

        self.max_length = 21

    def predict_sentiments(self, tweets):
        messages = self.tokenizer.texts_to_sequences(tweets)
        messages = pad_sequences(messages, padding='post', maxlen=self.max_length)

        predictions = self.model.predict(messages)

        sentiments = []

        for prediction in predictions:
            indice = np.argmax(prediction)
            probability = prediction[indice]
            valid_prediction = probability > 0.8

            result = "Not sure"
    
            if valid_prediction:
                if indice == 1:
                    result = "Positive"
                else:
                    result = "Negative"

            sentiments.append(result)

        return sentiments
