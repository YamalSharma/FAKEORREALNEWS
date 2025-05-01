


import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

fake_df['label'] = 0  # 0 = fake
real_df['label'] = 1  # 1 = real

df = pd.concat([fake_df, real_df], axis=0)
df = df[['text', 'label']]
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle


def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.lower()
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single chars
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['text'] = df['text'].apply(clean_text)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))


def predict_news(news_text):
    cleaned = clean_text(news_text)
    vec = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vec)[0]
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"


st.title("📰 Fake News Classifier")

st.write("""
    ### Check if a news article is **Fake** or **Real**!
    Enter a news article or headline below, and the model will classify it for you.
""")

user_input = st.text_area("Enter news text:")

if st.button("Classify News"):
    if user_input:
        result = predict_news(user_input)
        st.write("Prediction:", result)
    else:
        st.write("Please enter some text!")

