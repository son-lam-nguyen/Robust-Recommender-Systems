# Robust-Recommender-Systems
Integrating Shapley Value-driven Valuation (SVV) and Aspect-Based Sentiment Analysis (ABSA-BERT)

This repository contains the source code, datasets, and results for the research project:
“Integrating Shapley Value-driven Pruning and Aspect-Based Sentiment Analysis for Robust Recommender Systems.”
The project aims to build a robust, sentiment-aware recommender system that combines data pruning (SVV) and aspect-level sentiment analysis (ABSA-BERT) to improve accuracy, robustness, and explainability.

🧠 1. Project Overview

Traditional recommender systems often suffer from noisy data, biased interactions, and limited interpretability.
This project proposes a hybrid framework that integrates:

SVV (Shapley Value-driven Valuation): quantifies and removes low-value interactions to reduce noise and improve robustness.

ABSA-BERT: extracts aspect-level sentiment signals from reviews to enhance semantic understanding.

By fusing these two methods, the system generates accurate, reliable, and explainable recommendations — particularly effective in the tourism and hospitality domains where user feedback is emotional and diverse.

💾 2. Datasets
Dataset	Description	Source:

Google Local Reviews:	Reviews and ratings of hotels and restaurants from Google Maps, used to predict ratings.	

Hotel Recommendation Dataset:	User–item interactions and hotel reviews for tourism recommendations.	

https://github.com/akshayreddykotha/rating-prediction-google-local
https://www.kaggle.com/datasets/hariwh0/hotelrec-dataset-1
