# Customer Churn Prediction

This project predicts customer churn using the well-known **Telco Customer Churn** dataset from IBM.  
It demonstrates a complete end-to-end data science workflow including **EDA, preprocessing, model training, evaluation, explainability, and a production-style prediction pipeline**.

---

## 📌 Problem Statement
Customer churn is one of the biggest revenue leaks for subscription-based businesses.  
The objective of this project is to **build a machine learning model that predicts whether a customer will churn**, enabling proactive retention strategies.

---

## 🗂 Dataset
**Source:** IBM / Kaggle  
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset includes:
- Customer demographics  
- Account info (tenure, contract type, payment method)  
- Subscription features (phone, internet services)  
- Billing info  
- **Churn label** (Yes/No)

Total Rows: ~7043

---

## 🧹 1. Data Cleaning & Preprocessing
Key steps:
- Convert `TotalCharges` from string → numeric  
- Handle missing values in `TotalCharges`  
- Encode categorical variables using **One-Hot Encoding**  
- Scale/clean numeric fields  
- Train-test split (80/20, stratified)

---

## 📊 2. Exploratory Data Analysis (EDA)
Explored:
- Churn distribution  
- Tenure vs churn  
- Monthly charges vs churn  
- Contract type impact  
- Payment method effect  
- Service combinations and churn  
- Correlation heatmaps

Key Insights:
- Month-to-month contracts have the highest churn  
- Fiber-optic internet has higher churn  
- High monthly charges correlate with churn  
- Customers with long tenure churn less

---

## 🤖 3. Models Trained & Compared
Three models were evaluated:

### **1. Logistic Regression**
Baseline linear model.

### **2. Random Forest**
Strong non-linear tree-based model.

### **3. XGBoost**
Gradient-boosted trees → **best performer**.

Performance metrics:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC

XGBoost achieved the highest **ROC-AUC**, indicating excellent classification ability.

---

## 🔍 4. Explainability (SHAP)
SHAP was used to understand:
- **Global feature importance**
- **Feature impact on model output**
- **Customer-level churn explanations**

Insights:
- Contract type is the biggest driver of churn  
- MonthlyCharges strongly increase churn probability  
- Tenure is negatively correlated with churn  
- Fiber-optic internet service increases churn risk

---

## 🚀 5. Production-Style Prediction Pipeline
A full ML pipeline was built using:
- `ColumnTransformer`  
- `OneHotEncoder`  
- `XGBClassifier`  
- `Pipeline`  

Includes a helper function:

predict_single_customer()

This function:

Accepts raw customer data as a dictionary

Runs preprocessing + model

Outputs churn probability and label

The final model is saved as:

models/churn_pipeline_xgb.pkl

## 📁 Project Structure

```telco-churn-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
├── src/
├── models/
├── reports/
├── README.md
└── requirements.txt
```

## 📈 Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | ~80%     | Moderate  | Low    | Moderate | ~0.84   |
| Random Forest        | ~85%     | Good      | Good   | Good     | ~0.89   |
| XGBoost              | ~86%     | High      | High   | High     | ~0.91   |


XGBoost selected as final model.

## 🏁 Conclusion
This project demonstrates a real-world telecom churn pipeline with:

Solid feature engineering

Strong machine learning modeling

Explainability with SHAP

Production-style pipeline

Clean, modular folder structure

Perfect for resumes, portfolios, and interviews.

## 📌 Technologies Used

Python

Pandas

NumPy

Scikit-Learn

XGBoost

SHAP

Matplotlib / Seaborn

Jupyter Notebook
