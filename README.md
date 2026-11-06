# Customer Churn Prediction

This machine learning project that explores customer churn patterns and builds classification models to identify customers who are likely to leave a telecommunications company.

The project uses the IBM Telco Customer Churn dataset to investigate customer behaviour, compare different machine learning algorithms, and evaluate how churn predictions can support business retention strategies.

## Problem Statement

1. **Limited Churn Visibility**

   Telecommunications companies lack sufficient insight of how many customers are at risk of leaving their services.

3. **Revenue and Business Impact**

   Customer churn will result in revenue losses, higher customer acquisition costs (CAC), and negative impacts on business growth.

5. **Lack of Early Risk Identification**

   Without having a proper insight, companies will struggle to prepare and implement retention strategies before customers leave.

## Dataset

This project uses the **IBM Telco Customer Churn dataset** containing customer information from a telecommunications company.

### Dataset Overview

- **Total records:** 7,043 customers
- **Total columns:** 21
- **Target variable:** `Churn`
- **Classification type:** Binary classification

### Target Distribution

| Churn Status | Customers |
|---|---:|
| No | 5,174 |
| Yes | 1,869 |

The target variable indicates whether a customer has left the telecommunications company.

The dataset includes customer demographics, subscribed services, account information, contract types, payment methods, and billing details.

### Dataset Source

- **Platform:** Kaggle
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Uploader:** Blastchar

The dataset is used for educational and machine learning experimentation.

## Data Cleaning & Exploratory Data Analysis

### Data Cleaning

The following preprocessing steps were performed:

- Checked for duplicate records.
- Identified 11 missing values in `TotalCharges`.
- Converted `TotalCharges` from string to numeric format.
- Replaced the missing `TotalCharges` values with 0, as all affected customers had a tenure of 0 months.
- Excluded `customerID` from the machine learning features because it serves as an identifier rather than a meaningful predictive feature.

### Exploratory Data Analysis

Investigated relationships between customer churn and several features:

- **Contract:** Month-to-month customers had a churn rate of 42.7%, compared to 11.3% for one-year contracts and 2.8% for two-year contracts.
- **Tenure:** Churned customers had a median tenure of 10 months, compared to 38 months for customers who stayed.
- **Internet Service:** Fiber optic customers had a churn rate of 41.89%, higher than DSL customers (18.96%).
- **Monthly Charges:** Churned customers had higher average monthly charges ($74.44) than customers who stayed ($61.27).

From the analysis, contract type, customer tenure and internet service appear to have noticeable relationships with customer churn.

## Machine Learning Methodology

### Model Selection

Three classification algorithms were tested to compare their performance in predicting customer churn:

- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

Rather than assuming one algorithm would perform best, all three were evaluated to identify which performed better for the project's objective.

### Data Preprocessing

Before training, a preprocessing pipeline was created using:

- **StandardScaler:** Standardised numerical features to ensure they were on a comparable scale.
- **OneHotEncoder:** Converted categorical features into numerical representations without introducing an artificial hierarchy.

A `ColumnTransformer` and `Pipeline` were used to organise preprocessing and model training while preventing preprocessing leakage between training and testing data.

### Hyperparameter Tuning

`GridSearchCV` was used to test different hyperparameter combinations through 5-fold cross-validation.

Different scoring metrics were explored, including precision and recall, depending on the optimisation objective.

### Threshold Tuning

After selecting Logistic Regression as the preferred model, its classification threshold was adjusted using validation data.

A threshold of **0.55** was selected because it provided approximately 74% precision while maintaining at least 40% recall.

The final model was then evaluated on the held-out test set.

## Model Evaluation & Results

### Model Comparison

The three models were evaluated using validation data to determine which performed best.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 81.72% | 71.63% | 51.51% | 59.92% |
| Decision Tree | 79.33% | 66.02% | 45.48% | 53.86% |
| Random Forest | 80.75% | 68.98% | 49.83% | 57.86% |

Logistic Regression achieved the highest scores across all four evaluation metrics. It was selected as the preferred model for further evaluation.

### Threshold Selection

A classification threshold of **0.55** was selected using validation data.

This threshold achieved approximately 74% precision while maintaining at least 40% recall, matching the project's objective of prioritising precision without allowing recall to drop too low.

### Final Test Results

The final Logistic Regression model was evaluated on the test set using the selected threshold.

| Metric | Result |
|---|---:|
| Accuracy | 79.77% |
| Precision | 69% |
| Recall | 44% |
| F1-score | 54% |

The model correctly identified 69% of the customers it predicted would churn. However, it only identified 44% of all actual churners.

### Confusion Matrix

| | Predicted: No Churn | Predicted: Churn |
|---|---:|---:|
| Actual: No Churn | 960 | 75 |
| Actual: Churn | 210 | 164 |

The model produced 164 true positives and 210 false negatives. This means a significant number of customers who eventually churned were not identified by the model.

### Conclusion

Although Logistic Regression performed better than the other tested models, I consider its 44% recall insufficient for practical deployment.

For this project, I would prefer to achieve at least 50% recall while maintaining reasonable precision.

The model is therefore treated as a prototype rather than a production-ready solution. Further improvements, testing with more suitable historical data, and evaluation in a real retention campaign would be necessary before deployment.
