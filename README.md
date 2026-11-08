# ML Classification Streamlit Application

This repository hosts an interactive Streamlit application that evaluates and compares 5 Machine Learning classification models.

## a. Problem Statement
The objective of this project is to develop a robust machine learning workflow that automatically classifies synthetic data points into binary targets using 15 informative and redundant numerical features. The application provides an interactive web interface for both batch evaluation and real-time single-instance prediction across 5 distinct classification models.

## b. Dataset Description
- **Total Instances:** 1000 observations (split 70% train / 30% test).
- **Total Features:** 15 dimensions (10 highly informative features, 5 redundant features to simulate realistic noise).
- **Target Variable:** Binary classification label (0 or 1).

## c. GitHub Repository Link
- **Repository URL:** [https://github.com/your-username/ml-classification-dashboard](https://github.com/your-username/ml-classification-dashboard)

## d. Models Used & Comparison Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8467 | 0.9191 | 0.8581 | 0.8355 | 0.8467 | 0.6936 |
| **Decision Tree** | 0.8300 | 0.8304 | 0.8531 | 0.8026 | 0.8271 | 0.6614 |
| **K-Nearest Neighbor (KNN)** | 0.9167 | 0.9774 | 0.9635 | 0.8684 | 0.9135 | 0.8377 |
| **Naive Bayes** | 0.8467 | 0.9029 | 0.8955 | 0.7895 | 0.8392 | 0.6988 |
| **Support Vector Machine (SVM)** | 0.9467 | 0.9902 | 0.9416 | 0.9539 | 0.9477 | 0.8934 |

### Observations on Model Performance

- **Logistic Regression:** Performs well as a baseline with an accuracy of **84.67%** and a solid AUC of **0.9191**. It exhibits balanced precision (85.81%) and recall (83.55%), showing that the dataset's features are linearly separable to a good extent.
- **Decision Tree:** Lowest performing model overall with an accuracy of **83.00%** and AUC of **0.8304**. It shows signs of slight overfitting as it struggled compared to neighborhood and boundary-based methods.
- **KNN:** Excellent performance with an accuracy of **91.67%** and a high AUC of **0.9774**. It has the highest precision (**96.35%**), demonstrating that local neighborhood patterns are highly representative in this multi-feature space.
- **Naive Bayes:** Achieves **84.67%** accuracy, matching Logistic Regression. However, it shows a lower recall (**78.95%**) and a lower AUC (**0.9029**), indicating that the feature independence assumption slightly limits its classification confidence.
- **Support Vector Machine:** Outstanding top-tier performance with an accuracy of **94.67%**, AUC of **0.9902**, and an MCC of **0.8934**. It maintains both high precision (94.16%) and the highest recall (95.39%).

### Overall Winner for the Dataset
🏆 **Support Vector Machine (SVM)**  
**Reason:** SVM is the clear winner across almost every single metric. It reached the highest Accuracy (94.67%), AUC Score (0.9902), F1 Score (94.77%), and MCC Score (0.8934). It effectively maps the 15-dimensional boundary to maximize separation, proving to be the most robust classifier for this dataset.

---
## Repository Structure
- `app.py`: The entry point for the Streamlit dashboard.
- `requirements.txt`: Python package requirements.
- `test_data.csv`: Sample generated testing data with 15 features.
- `model/`: Folder containing saved trained classifiers (.joblib).

## How to Run Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`
