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
https://github.com/akshayreddykotha/rating-prediction-google-local

Hotel Recommendation Dataset:	User–item interactions and hotel reviews for tourism recommendations.	
https://www.kaggle.com/datasets/hariwh0/hotelrec-dataset-1

⚙️ 3. Environment Setup
Requirements:
  - Python 3.10+
  - PyTorch 2.0+
  - Transformers 4.40+
  - FastSHAP
  - NumPy, Pandas, Scikit-learn, Matplotlib

🧩 4. Baseline Implementations
Each baseline model is implemented as a separate Python file for fair and modular comparison.

- BPRMF Baseline.py:	Implements Bayesian Personalized Ranking Matrix Factorization – collaborative filtering baseline.
- LightGCN Baseline.py:	Graph-based collaborative model using simplified GCN layers for recommendations.
- AE Baseline.py: AutoEncoder-based collaborative filtering baseline.
- DAE Baseline.py:	Denoising AutoEncoder, used as the base model for SVV training.
- DeepCoNN Baseline.py: Deep Cooperative Neural Network combining user and item text reviews.
- SVV Baseline.py: Implements Shapley Value-driven Valuation (SVV) for noise pruning and robustness.
- ABSA-BERT Baseline.py: Performs Aspect-Based Sentiment Analysis using BERT to extract fine-grained sentiment embeddings.
- SVV and ABSA-BERT Baseline.py:	Final hybrid model (ours) combining both SVV and ABSA-BERT for robust and explainable recommendations.

Run Baselines: 
- python "BPRMF Baseline.py"
- python "LightGCN Baseline.py"
- python "AE Baseline.py"
- python "DAE Baseline.py"
- python "DeepCoNN Baseline.py"
- python "SVV Baseline.py"
- python "ABSA-BERT Baseline.py"
- python "SVV and ABSA-BERT Baseline.py"

Each script:
- Loads the dataset
- Trains the model
- Evaluates using Recall@K, NDCG@K, AUC, RMSE
- ...
  
📊 5. Evaluation Metrics
- Recall@K: Measures how many relevant items appear in the top-K recommendations.
- NDCG@K: Evaluates the ranking quality and order of recommendations.
- MRR@K: Measures ranking accuracy using the first relevant item.
- Accuracy@K: Fraction of correctly predicted items within top-K.
- AUC: Measures model discrimination between positive and negative samples.
- RMSE: Measures deviation between predicted and true ratings (lower is better).
