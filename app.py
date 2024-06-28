import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import re
import nltk
import pickle
import streamlit as st
import requests

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

folders = ["fake_news", "job_scams", "phishing", "political_statements", "product_reviews", "sms", "twitter_rumours"]

def loadTfidf(fromFile):
    filehandle = open(fromFile,'rb')
    clf = pickle.load(filehandle)
    return clf

def loadModel(fromFile):
    filehandle = open(fromFile,'rb')
    model = pickle.load(filehandle)
    return model

def clean_and_lemmatize_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_article(article):

    # if category=='fake_news':
    #     model:Pipeline = loadModel('models/fn.pkl')
    #     return model.predict([article])[0]

    tfidf:TfidfVectorizer = loadTfidf(f'models/{category}_ftfidf.pkl')
    clf:VotingClassifier = loadModel(f'models/{category}_clf.pkl')
    cleaned_article = clean_and_lemmatize_text(article)
    article_tfidf = tfidf.transform([cleaned_article]).toarray()
    prediction = clf.predict(article_tfidf)
    # return 'Fake' if prediction == 1 else 'Real'
    return prediction

# Function to call the API
def call_api(text, category):
    # Replace 'YOUR_API_URL' with your actual API endpoint
    # api_url = 'https://your-api-url.com/endpoint'
    payload = {'text': text, 'category': category}
    # phishing_tfidf.pkl
    # response = requests.post(api_url, json=payload)
    response = predict_article(text)
    print(text, response)
    return response
    # if response.status_code == 200:
    #     return response.json().get('result')  # Assuming API returns {'result': 0} or {'result': 1}
    # else:
    #     st.error("API call failed")
    #     return None

# Streamlit app layout
st.title("Text Classification App")

# Text input field
user_input = st.text_area("Enter your text (max 5000 characters):", max_chars=5000)

# Dropdown selector
category = st.selectbox(
    "Select a category:",
    folders
)

# Button to make the API call
if st.button("Submit"):
    if user_input and category:
        with st.spinner("Calling the API..."):
            result = call_api(user_input, category)
            if result is not None:
                if result == 0:
                    st.image("real.gif")
                elif result == 1:
                    st.image("fake.gif")
                else:
                    st.error("Unexpected result from the API")
            else:
                st.error("No result from the API")
    else:
        st.warning("Please enter text and select a category.")

# To run this app, save the code in a file named `app.py` and run `streamlit run app.py` in your terminal.