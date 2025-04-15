# MSBA-6421 - Practice-Predictive-Analytics  

## 📌 About This Repository  
This repository is a collection of Jupyter Notebooks documenting my **hands-on learning** of various **Machine Learning techniques**. Each notebook explores a different ML concept with practical implementation.  

---

# 📒 Notebooks Summary – Predictive Analytics Projects

---

## 📘 Notebook 1: Feature Selection & Hyperparameter Tuning

This notebook dives into the nitty-gritty of fine-tuning model performance through careful feature selection and hyperparameter optimization.

### Techniques Covered:
- **Recursive Feature Elimination (RFE)** – Prunes your features like a bonsai master, keeping only what matters.
- **GridSearchCV** – Hyperparameter tuning made lazy — let the machine try all combinations for you.
- **K-Fold Cross-Validation** – Gives your model a real stress test across multiple data splits, ensuring it’s not just good on one lucky fold.

---

## 📙 Notebook 2: Ordinal Variable Treatment & Model Performance Analysis

Explores how to handle ordinal variables and their impact on model performance — numeric vs categorical — and goes deep on evaluation.

### Techniques Covered:
- **Numeric vs Categorical Treatment** – Side-by-side comparison of different encoding strategies for ordinal variables. Because how you treat your data matters.
- **Nested Cross-Validation** – The gold standard for model evaluation, nesting CV loops like a Russian doll.
- **Learning Curve Analysis** – Plots that tell you if your model’s getting smarter or just memorizing flashcards.
- **GridSearchCV** – Once again, because good tuning deserves repetition.
- **Model Evaluation** – Final testing on a 20% holdout dataset to check if your model generalizes or flops under pressure.

---

## 📕 Notebook 3: Predicting Customer Spending 

This notebook tackles customer spending prediction using a variety of regression techniques, applied to both the full dataset and a restricted subset where purchases actually occurred. The goal: model how much a customer is likely to spend — if anything — and compare modeling strategies across scenarios.

### Techniques Covered:
- **Linear Regression, k-NN, Regression Trees, SVM Regression, Neural Networks, and Ensembles** – A buffet of models to see what works best.
- **Hyperparameter Tuning** – Tweaking knobs like a DJ to get the cleanest predictive sound.
- **Feature Normalization** – Keeping it fair across scales, especially for distance-based models.
- **Comparison of Full vs Purchase-Only Data** – Analyzing performance differences when modeling for all customers vs only buyers.
- **Model Interpretation & Evaluation** – Talking metrics: RMSE, MAE, R², and which models actually generalize well.

🎯 **Outcome:** Insights into which algorithms perform best for numeric prediction in noisy real-world data, and how model behavior shifts when restricted to actual buyers.

---

## 📗 Notebook 3: Spam Detection & Cost-Sensitive Classification 

This notebook focuses on spam email classification, exploring both plain accuracy-focused models and those built with **cost sensitivity** in mind (because in the real world, not all mistakes cost the same).

### Techniques Covered:
- **Multiple Classification Models** – Logistic Regression, k-NN, Naive Bayes, Decision Trees, SVM, Random Forests, etc. (aka the whole squad).
- **Hyperparameter Tuning** – Optimizing model configs via GridSearchCV and cross-validation.
- **Normalization** – Because the dataset is spicy with varying scales.
- **Cost-Sensitive Learning** – Using a 10:1 misclassification cost ratio (false negatives are way more expensive).
- **Model Evaluation with Metrics** – Accuracy, Precision, Recall, F1, AUC, Misclassification Cost.
- **Visualizations** – ROC curves, lift charts, confusion matrices—bringing model performance to life.
- **Nested Cross-Validation** – Preventing overfitting while tuning for real-world cost impact.

🎯 **Outcome:** A deep dive into not just which models perform well, but which ones are most cost-effective when it matters most (like avoiding missing a spam bomb in your inbox).

---

📌 *More notebooks to come as I keep exploring this wild world of predictive modeling!*

