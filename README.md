# Twitter-Sentiment-Analysis

A machine learning project that analyzes sentiments in tweets using Logistic Regression on the Sentiment140 dataset. Built entirely in **Google Colab**, this project demonstrates preprocessing, model training, evaluation, and saving the model using `pickle`.

---

### 📂 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Model Performance](#-model-performance)
- [How to Run](#-how-to-run)
- [Results & Visualizations](#-results--visualizations)
- [Improvements](#-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

### 🧠 Overview

This project applies NLP and machine learning to perform binary sentiment classification (positive/negative) on a dataset of 1.6 million tweets. The final model was trained using **Logistic Regression** and serialized using the `pickle` module for future reuse.

---

### ✅ Features

- Preprocessing tweets (removing usernames, links, punctuations, stopwords, etc.)
- Feature extraction using TF-IDF
- Logistic Regression model for classification
- Accuracy score and confusion matrix
- Model serialization with pickle
- Entire project runs on Google Colab

---

### 🛠️ Tech Stack

- Python (Google Colab)
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- NLTK
- pickle

---

### 📊 Dataset

- **Source**: [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/kazanova/sentiment140)
- **Size**: 1.6 million labeled tweets
- **Labels**:  
  - `0`: Negative  
  - `4`: Positive (converted to `1`)

---

### 🚀 Model Performance

| Metric       | Score   |
|--------------|---------|
| Train Accuracy | 79.87% |
| Test Accuracy  | 77.67% |

---

### 🖥️ How to Run

> ✅ No installation needed!

1. Open the notebook in Google Colab:  
   [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sZiXBKMhKJN_fm5c7oUMgYFSBQP935DN)

2. Follow these steps inside the notebook:
   - Load and preprocess the data
   - Vectorize text using TF-IDF
   - Train the Logistic Regression model
   - Evaluate the model
   - Download the trained model (optional)

---

### 📈 Results & Visualizations

- Word clouds for positive and negative tweets
- Accuracy score and classification metrics
- Sample predictions

---

### 💡 Improvements

- Add support for neutral sentiment (multi-class classification)
- Deploy a web interface using Flask or Streamlit
- Integrate Twitter API for real-time analysis
- Experiment with advanced models like Naive Bayes, SVM, or LSTM

---

