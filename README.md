# Sentiment Analysis of Game Review

This project focuses on the sentiment analysis of game reviews, particularly focusing on user-generated reviews. The goal is to classify these reviews as positive or negative, helping game developers and marketers understand user feedback more effectively.

## Overview

In the gaming industry, user reviews are a crucial source of feedback that can significantly influence a game's success. This project leverages natural language processing (NLP) techniques to analyze and classify these reviews. By automatically identifying the sentiment behind each review, the model can assist in summarizing large volumes of user feedback, which is often challenging to process manually.

Data for Training: https://www.kaggle.com/datasets/andrewmvd/steam-reviews/data

This study was conducted using [Google Colaboratory](https://colab.research.google.com/)

## Key Features

- **Data Collection**: The training dataset consists of Steam game reviews obtained from Kaggle. This dataset was chosen for its comprehensive representation of user sentiment within the gaming community.
- **Preprocessing**: The text data underwent extensive cleaning and preprocessing, including stopword removal, stemming, and lemmatization. This step was crucial for optimizing the model's ability to accurately classify sentiment.
- **Modeling**: Several machine learning models were trained to classify the sentiment of the reviews, including logistic regression, random forest, and naive Bayes. Among these, logistic regression demonstrated the best performance, making it the preferred model for this project.
- **Performance**: The logistic regression model achieved strong performance with 81% accuracy on validation data, validating its effectiveness in analyzing and classifying game reviews. This model can be used by developers and marketers to gain actionable insights from user feedback.
- **Testing**: The trained logistic regression model was further tested on reviews scraped from the Google Play Store, ensuring its applicability to a broader range of game reviews across different platforms. The model achieved 80% accuracy with the scraped data.
- **Application**: It could highlight how game developers, marketers, or data analysts might use the sentiment analysis results to make informed decisions, such as improving game features, addressing common user complaints, or tailoring marketing campaigns.
