#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:26:34 2021

@author: sharad mittal
"""

import pandas as pd
import numpy as np
import pickle 
import re
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

#label encoder
le = LabelEncoder()

#Pickled model
pickle_in=open('language_predictor.pkl','rb')
language_predictor=pickle.load(pickle_in)

#Count vectorizer
cv_pickle=open('vectorize.pkl','rb')
cv=pickle.load(cv_pickle)

              
def lang_predict(text):
     # loading the dataset
     data = pd.read_csv("Language Detection.csv")
     y = data["Language"]
     # label encoding
     y = le.fit_transform(y)
     #Cleaning the input text
     text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]','', text)
     text = re.sub(r'[[]]', '', text)
     text = text.lower()
     data = [text]
     # converting text to bag of words model (Vector)
     x = cv.transform(data).toarray() 
     # predicting the language
     lang = language_predictor.predict(x)
     # finding the language corresponding the the predicted value
     lang = le.inverse_transform(lang) 
     # return the predicted language
     return lang[0] 

def main():
    st.title("Language Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Language Predictor using NLP</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text=st.text_input("Text to Predict","Type Here")
    result=""
    if st.button("Predict"):
        result=lang_predict(text)
    st.success('The given text is written in {}'.format(result))
    if st.button("About"):
        st.text("Predicting Language of a given text using NLP")
        st.text("API built with Streamlit")
        

if __name__== '__main__':
    main()
