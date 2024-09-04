# -*- coding: utf-8 -*-
"""
@author: Omkar Nallagoni11
"""
import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from PIL import Image


pickle_in = open("nbc_model.pkl","rb")
classifier=pickle.load(pickle_in)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

tf_idf = TfidfVectorizer()
ps = PorterStemmer()
corpus = set()

def preprocess(text):
    
    ## removing unwanted space
    text = text.strip()
    
    ## removing html tags 
    text = re.sub("<[^>]*>", "",text)
    
    ## removing any numerical values
    text = re.sub('[^a-zA-Z]', ' ',text)
    
    ## lower case the word
    text = text.lower()
    
    text = text.split()
    
    ## stemming the word for sentiment analysis do not remove the stop word
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text

def main():
    st.title('Sentiment Analysis')
    st.write("Enter a review below to predict its sentiment.")

    result=''
    text=st.text_input('Enter the review','Type Here')
    if st.button('Predict'):
        if text:
            processed_text=preprocess(text)
            processed_text=tfidf.transform([processed_text])
            prediction=classifier.predict(processed_text)
            st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
        else:
            st.write('ENter the rreview')

    

if __name__=='__main__':
    main()
    
    
    