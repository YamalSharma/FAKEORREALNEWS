# Fake News Classifier with Streamlit

## Project Overview:
This project implements a **Fake News Classifier** using Natural Language Processing (NLP) and **Machine Learning**. The classifier distinguishes between real and fake news articles based on their content. The app utilizes **Streamlit** to provide a simple user interface, allowing users to input news articles and get predictions.

## Features:
- **Text Preprocessing**: Removes punctuation, stopwords, and unnecessary characters to clean the news text.
- **TF-IDF Vectorization**: Converts text data into numerical form for use in machine learning.
- **Logistic Regression Model**: The model is trained on a dataset of fake and real news articles.
- **Streamlit UI**: A user-friendly interface to input news text and display predictions.
- **Accuracy**: Achieves an accuracy of over 98%, providing reliable predictions.

## Technologies Used:
- **Python**: Core programming language.
- **NLTK**: Used for text preprocessing (removing stopwords, tokenizing).
- **scikit-learn**: Used for machine learning (Logistic Regression, TF-IDF Vectorization, and Evaluation).
- **Streamlit**: Used to build the web interface for the app.
- **Pandas** & **NumPy**: Used for data processing and handling.

## Dataset:
The dataset used consists of two CSV files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.
- The dataset was sourced from [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).
# How to Run Fake News Classifier with Streamlit

This guide provides step-by-step instructions to run the **Fake News Classifier** project locally.

## Prerequisites:
Ensure that you have the following installed on your local machine:
- **Python 3.x**: If you don't have Python installed, download it from [Python Official Site](https://www.python.org/downloads/).
- **Git**: If Git is not installed, get it from [Git Official Site](https://git-scm.com/).

## Steps to Run the Project:

# 1. Clone the Repository:
    git clone https://github.com/YamalSharma/FAKEORREALNEWS.git
    cd FAKEORREALNEWS

# 2. Install Dependencies:
    pip install pandas
    pip install numpy
    pip install scikit-learn
    pip install streamlit
    pip install nltk


# 3. Download the Dataset:
  Download the dataset from Kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
   After downloading, place the 'Fake.csv' and 'True.csv' files in the project directory (FAKEORREALNEWS).

# 4. Run the Streamlit App:
streamlit run fake_news_classifier_ui.py

# 5. Access the App:
   Once the app is running, open your web browser and go to:
  http://localhost:8501
  You can now input news articles and classify them as either Fake or Real.

# Model Evaluation:
  The classifier achieves an accuracy of over 98%.
  Detailed precision, recall, and F1-score metrics are available in the classification report.


# Troubleshooting:
  - Ensure all dependencies are installed correctly and you're running Python 3.x.
  - Make sure the dataset is placed in the correct directory (FAKEORREALNEWS).

