# NLP Sentiment Classifier
Machine learning project for classifying product reviews into Positive, Negative, and Neutral sentiments using TF-IDF and multiple classification models.

## Project Overview
The goal of this project is to apply Natural Language Processing (NLP) techniques to analyze customer reviews and predict their sentiment, The text data is preprocessed and transformed into numerical features using TF-IDF vectorization, Different classification models are trained and evaluated to compare their performance.

## Dataset
The dataset contains product reviews along with their corresponding sentiment labels.

Main columns used:
- Review / Summary / Rate (text data)
- Sentiment (target label)

## Data Preprocessing

The following preprocessing steps were applied:
- Cleaning text data
- Converting text to lowercase
- Removing stop words
- Removing dublicate rows

## Models Used

The following machine learning models were implemented:

- Decision Tree Classifier
- Multinomial Naive Bayes
- Logistic Regression

## Model Evaluation

The dataset was split into:
- 80% Training set
- 20% Test set

Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## How to Run the Project

1. Clone the repository
2. Install the required libraries:
