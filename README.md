# Customer Churn Prediction – End-to-End Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)


## 📌 Overview
This project demonstrates an **end-to-end machine learning workflow** for predicting customer churn in a subscription-based business context.  
The focus is not only on model performance, but also on **clean pipelining, reproducibility, explainability, and stakeholder-ready reporting**.

The repository is structured to reflect how a real data science project would be developed, reviewed, and extended in a professional environment.

---

## Business Problem
Customer churn directly impacts revenue and growth.  
The goal of this project is to:
- Predict whether a customer is likely to churn
- Identify **key drivers of churn**
- Enable **data-driven retention strategies**

---

## What This Project Showcases
- Complete data science lifecycle (EDA → modeling → evaluation)
- **Production-style ML pipeline** using preprocessing + model chaining
- Multiple model comparison (Logistic Regression, Random Forest, XGBoost)
- Use of **appropriate churn metrics** (ROC-AUC, Precision, Recall, F1)
- **Model explainability using SHAP**
- A **professional PDF report** summarizing insights for non-technical stakeholders

---

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

**Includes a helper function:**

`predict_single_customer()`

This function:

- Accepts raw customer data as a dictionary

- Runs preprocessing + model

- Outputs churn probability and label

The final model is saved as:

`models/churn_pipeline_xgb.pkl`

---

## 📈 Results Summary

| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | ~80%     | Moderate  | Low    | Moderate | ~0.84   |
| Random Forest        | ~85%     | Good      | Good   | Good     | ~0.89   |
| XGBoost              | ~86%     | High      | High   | High     | ~0.91   |


- **XGBoost** delivered the best overall performance and was selected as the final model.

---

## 🏁 Conclusion
This project demonstrates a real-world telecom churn pipeline with:

- Solid feature engineering

- Strong machine learning modeling

- Explainability with SHAP

- Production-style pipeline

- Clean, modular folder structure

---

## 📌 Technologies Used

- Python

- Pandas

- NumPy

- Scikit-Learn

- XGBoost

- SHAP

- Matplotlib / Seaborn

- Jupyter Notebook

---

## 🛠️ How to Run
```bash
pip install -r requirements.txt
python src/train.py
```

## Author
Pankaj Kapri
Data Science | Machine Learning | End-to-End ML Pipelines

---

## 📄 License

This project is licensed under the MIT License.
