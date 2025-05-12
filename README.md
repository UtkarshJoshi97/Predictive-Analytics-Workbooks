### 📒 Detailed Notebook Summaries
---
### 01_Notebook [01_Feature_Selection_Hyperparameter_Tuning.ipynb](01_Feature_Selection_Hyperparameter_Tuning.ipynb)

This notebook focuses on improving model performance through **feature selection** and **hyperparameter tuning** using classical machine learning models.

##### Key Work:
- Implemented **Recursive Feature Elimination (RFE)** to identify the most impactful features.
- Applied **GridSearchCV** to optimize hyperparameters for algorithms like Logistic Regression and SVM.
- Used **K-Fold Cross-Validation** to evaluate model stability across different splits.
- Compared model performance before and after tuning, noting improvements in accuracy and generalization.

 **Objective:** Build a strong foundation by selecting only relevant features and tuning models to avoid underfitting or overfitting.
---

### 02_Notebook [02_Ordinal_Treatment_Model_Performance.ipynb](02_Ordinal_Treatment_Model_Performance.ipynb)

This notebook investigates how **ordinal variables** (features with a natural order) affect model performance, and experiments with different treatments to handle them effectively.

##### Key Work:
- Tested both **numeric encoding** and **categorical encoding** strategies for ordinal features.
- Ran models like Decision Trees and Random Forests using both encodings to compare performance.
- Used **Nested Cross-Validation** to tune hyperparameters and prevent data leakage during evaluation.
- Visualized **learning curves** to diagnose under/overfitting.
- Evaluated models on a **20% holdout set** using metrics like accuracy and F1-score.

 **Objective:** Understand the impact of encoding strategies on model behavior and identify the most stable, accurate approach for ordinal data.
---

### 03_Notebook [03_Customer_Spending_Prediction.ipynb](03_Customer_Spending_Prediction.ipynb)

This notebook builds regression models to predict how much a customer is likely to spend, based on their historical behavior and profile data.

##### Key Work:
- Trained multiple regression models: **Linear Regression, k-NN, Regression Trees, SVM, Neural Networks, and Ensemble Methods**.
- Performed **feature normalization** to ensure consistent performance across models.
- Ran models on both the **entire dataset** and a **filtered subset** (customers who made purchases).
- Compared model performance using metrics like **RMSE, MAE, and R²**.
- Analyzed differences in model accuracy across full vs. purchase-only datasets.

 **Objective:** Explore which regression algorithms best capture customer spending patterns, and how data filtering influences predictive performance.
---

### [Spam_Classification_CostSensitive.ipynb](04_Spam_Classification_CostSensitive.ipynb)

This notebook tackles **spam email detection**, with a special emphasis on **cost-sensitive classification**—because in real life, false negatives can be expensive.

##### Key Work:
- Built multiple classifiers: **Logistic Regression, k-NN, Naive Bayes, Decision Trees, SVM, and Random Forests**.
- Introduced a **10:1 cost ratio** for false negatives vs. false positives to reflect real-world priorities.
- Tuned models using **GridSearchCV** with cost-sensitive metrics.
- Normalized features for scale-sensitive algorithms.
- Evaluated models using **accuracy, precision, recall, F1-score, AUC**, and **cost-based metrics**.
- Visualized performance using **ROC curves**, **confusion matrices**, and **lift charts**.

 **Objective:** Design models that aren’t just accurate, but economically efficient—prioritizing the minimization of high-cost misclassifications.
---
### 04_Notebook [04_Shallow_vs_Deep_Neural_Network.ipynb](04_Shallow_vs_Deep_Neural_Network.ipynb)

This notebook compares the performance of **shallow vs deep neural networks** using synthetic data.

#### Key Work:
- Generated 120K data points from a non-linear function; split evenly into training and testing sets.
- Trained neural networks with **1, 2, and 3 hidden layers**.
- Evaluated models using **Mean Squared Error (MSE)** across different neuron counts.
- Visualized how **network depth** impacts performance and learning capacity.

 **Objective:** Analyze the trade-offs between shallow and deep architectures in capturing non-linear patterns.
---

### 🧠 [05_Predictive_ClosingPriceMovements_Project.ipynb](05_Predictive_ClosingPriceMovements_Project.ipynb)

A full-scale project built around the Kaggle competition **Optiver – Trading at the Close**, focused on predicting **short-term price movements** of Nasdaq-listed stocks using auction and order book data.

Link: https://www.kaggle.com/competitions/optiver-trading-at-the-close/overview

##### Project Overview:
- Modeled the **final 10 minutes of Nasdaq trading**, a period known for high volatility and critical pricing decisions.
- Integrated **order book data** and **auction book signals** to inform predictions.
- Employed **time-series modeling techniques** using the Kaggle-provided **Python API**, which simulates a real-time environment and prevents lookahead bias.
- Engineered features capturing **supply-demand dynamics**, price trends, and volume shifts.
- Trained regression models to predict **next-step price returns**, evaluated using **Mean Absolute Error (MAE)**.
- Aligned with Kaggle's submission framework to generate and export `submission.csv`.

🎯 **Objective:** Build a robust model to support smarter, faster pricing decisions during the volatile final moments of trading—mirroring the real-world challenges faced by quants and market makers.
