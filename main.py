import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load IMDB dataset 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

#load pre tarined model 
model = load_model('simple_rnn_model.h5')

#helper function 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])


def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2)+ 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review 

## streamlit app
import streamlit as st

st.title('IMDB Sentiment Analysis')
st.write('Enter  a movie to review to clasify it a  positive or negative')

#user imput
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    #make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    

 # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

