import pickle
import numpy as np
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, Response, render_template

app = Flask(__name__)
model = keras.models.load_model('sentiment_model_lstm_v1/')
MAX_LENGTH = 21

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']

    with open('tokenizer.pickle', 'rb') as handle:
    	tokenizer = pickle.load(handle) 
        
    message = tokenizer.texts_to_sequences([message])
    message = pad_sequences(message, padding='post', maxlen=MAX_LENGTH)

    predictions = model.predict(message)

    indice = np.argmax(predictions[0])
    probability = predictions[0][indice]
    valid_prediction = probability > 0.8
    print(f'predictions: {predictions} | indice - {indice} | probability - {probability}')

    result = "Not sure"
    
    if valid_prediction:
        if indice == 1:
            result = "Positive"
        else:
            result = "Negative"
    
    return render_template('result.html',prediction = result)

if __name__ == '__main__':
    app.run(debug=True)
