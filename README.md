### Detailed Notebook Summaries
---
# Foundation & Fundamentals  
#### 01_Notebook: Feature_Selection_Hyperparameter_Tuning

This notebook focuses on improving model performance through **feature selection** and **hyperparameter tuning** using classical machine learning models.

##### Key Work:
Implemented Recursive Feature Elimination (RFE) to select key features, optimized hyperparameters using GridSearchCV for models like Logistic Regression and SVM, and evaluated model stability with K-Fold Cross-Validation—ultimately improving accuracy and generalization through performance comparisons before and after tuning.

 **Objective:** Build a strong foundation by selecting only relevant features and tuning models to avoid underfitting or overfitting.
 
---

#### 02_Notebook: Ordinal_Treatment_Model_Performance

This notebook investigates how **ordinal variables** (features with a natural order) affect model performance, and experiments with different treatments to handle them effectively.

##### Key Work:
Explored multiple encoding strategies for ordinal features, testing numeric vs. categorical approaches with models like Decision Trees and Random Forests. Employed Nested Cross-Validation to fine-tune hyperparameters while guarding against data leakage, visualized learning curves to assess model bias-variance trade-offs, and validated final performance on a 20% holdout set using accuracy and F1-score.

 **Objective:** Understand the impact of encoding strategies on model behavior and identify the most stable, accurate approach for ordinal data.
 
---
# Deep Dives & Advanced Applications

#### 03_Notebook: Customer_Spending_Prediction and Spam_Classification_CostSensitive
#### Customer_Spending_Prediction
This notebook builds regression models to predict how much a customer is likely to spend, based on their historical behavior and profile data.

##### Key Work:
Trained a variety of regression models—including Linear Regression, k-NN, Regression Trees, SVM, Neural Networks, and Ensemble Methods—after applying feature normalization to standardize input scales. Evaluated performance on both the full dataset and a filtered subset of purchasing customers, comparing results using RMSE, MAE, and R² to uncover insights into model accuracy across different data segments.

 **Objective:** Explore which regression algorithms best capture customer spending patterns, and how data filtering influences predictive performance.
 
---

#### Spam_Classification_CostSensitive

This notebook tackles **spam email detection**, with a special emphasis on **cost-sensitive classification**—because in real life, false negatives can be expensive.

##### Key Work:
Developed multiple classification models—including Logistic Regression, k-NN, Naive Bayes, Decision Trees, SVM, and Random Forests—while applying feature normalization for scale-sensitive methods. Incorporated a 10:1 cost ratio to penalize false negatives more heavily, aligning model objectives with real-world priorities. Used GridSearchCV with cost-sensitive metrics for hyperparameter tuning, and assessed performance through accuracy, precision, recall, F1-score, AUC, and cost-based measures, supported by visual diagnostics like ROC curves, confusion matrices, and lift charts.

 **Objective:** Design models that aren’t just accurate, but economically efficient—prioritizing the minimization of high-cost misclassifications.
 
---
#### 04_Notebook: Shallow_vs_Deep_Neural_Network

This notebook compares the performance of **shallow vs deep neural networks** using synthetic data.

#### Key Work:
Generated 120,000 data points from a non-linear function and split the data evenly into training and testing sets to explore model learning behavior. Trained neural networks with varying architectures—using 1, 2, and 3 hidden layers—and systematically evaluated their performance using Mean Squared Error (MSE) across different neuron counts. Visualized the relationship between network depth and predictive accuracy to analyze how increased complexity impacts learning capacity and model effectiveness.

 **Objective:** Analyze the trade-offs between shallow and deep architectures in capturing non-linear patterns.
 
---
# Project

####  05_Predictive_ClosingPriceMovements_Project
##### Note: The ensemble modeling approach showcased in this repository reflects the part of the project I worked on. While the overall project involved contributions from the entire team—including tree-based baselines, time-split models, and LSTM—this repo specifically highlights the ensemble technique I developed. A combined version with all components will be made available separately.

A full-scale project built around the Kaggle competition **Optiver – Trading at the Close**, focused on predicting **short-term price movements** of Nasdaq-listed stocks using auction and order book data.

Link: https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview

##### Project Overview:
- Modeled the **final 10 minutes of Nasdaq trading**, a period known for high volatility and critical pricing decisions.
- Integrated **order book data** and **auction book signals** to inform predictions.
- Employed **time-series modeling techniques** using the Kaggle-provided **Python API**, which simulates a real-time environment and prevents lookahead bias.
- Engineered features capturing **supply-demand dynamics**, price trends, and volume shifts.
- Trained regression models to predict **next-step price returns**, evaluated using **Mean Absolute Error (MAE)**.
- Aligned with Kaggle's submission framework to generate and export `submission.csv`.

##### Modeling Approaches:
This project explored multiple modeling strategies to tackle the complexity of short-term price prediction:

1. Baseline Tree-Based Models
Used models like XGBoost and LightGBM to assess how well they handle missing values and capture auction dynamics.

2. Time-Split Modeling
Split data into pre- and post-300 seconds (based on availability of near/far prices), trained separate models on each, and combined predictions for better accuracy.

3. Sequential Modeling with LSTM
Applied LSTM to leverage temporal patterns using engineered time-based features—performed well on sequential structures.

4. Ensemble Modeling
Leveraged a **weighted ensemble** of three model types to improve prediction stability and accuracy:
  - **Tree-Based Models** (e.g., XGBoost, LightGBM): 50% weight — strong baseline with consistent performance.
  - **GRU (Gated Recurrent Unit)** Model: 30% weight — added sequential learning to capture temporal dynamics.
  - **Transformer-Based Model**: 20% weight — introduced attention mechanisms to capture dependencies across time steps.
 **Why Ensemble?**  
  Tree-based models were highly stable but struggled with sequential patterns. The GRU and Transformer filled this gap by modeling **temporal behavior**, while the ensemble approach balanced **robustness and time-awareness**.

**Objective:** Build a robust hybrid model that combines the strengths of multiple architectures to support accurate pricing decisions in a fast-moving, high-stakes trading window — just like real-world quant systems.
