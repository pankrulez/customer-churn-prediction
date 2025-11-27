import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📉",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = os.path.join("models", "churn_pipeline_xgb.pkl")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    return joblib.load(model_path)

@st.cache_data
def load_example_data():
    path = os.path.join("data", "processed", "telco_churn_clean.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

pipeline = load_model()
df_example = load_example_data()

st.title("📉 Telco Customer Churn Prediction")
st.write("Predict customer churn using the trained machine learning pipeline.")

# ---------- SIDEBAR ----------
mode = st.sidebar.selectbox(
    "Select mode",
    ["Single Customer Prediction", "Batch Prediction (CSV)", "About / Info"]
)

# ---------- HELPER: PREDICT ----------
def predict_churn(input_df: pd.DataFrame):
    proba = pipeline.predict_proba(input_df)[:, 1][0]
    label = int(proba >= 0.5)
    return proba, label

# ---------- SINGLE CUSTOMER MODE ----------
if mode == "Single Customer Prediction":
    st.subheader("Single Customer Prediction")

    with st.form("churn_form"):
        st.markdown("### Contract & Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        st.markdown("### Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        st.markdown("### Customer Profile")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        if df_example is None:
            st.error("Example dataset not found. Cannot build input row.")
            st.stop()

        # Take a template row with ALL required columns
        template = df_example.iloc[[0]].copy()

        # Drop target column if it exists
        for col in ["Churn", "churn", "ChurnLabel"]:
            if col in template.columns:
                template = template.drop(columns=[col])

        # Override only the fields we expose in the form
        template["Contract"] = contract
        template["PaperlessBilling"] = paperless_billing
        template["PaymentMethod"] = payment_method
        template["InternetService"] = internet_service
        template["OnlineSecurity"] = online_security
        template["OnlineBackup"] = online_backup
        template["TechSupport"] = tech_support
        template["tenure"] = tenure
        template["MonthlyCharges"] = monthly_charges
        template["TotalCharges"] = total_charges

        # Now template has ALL columns the pipeline expects
        proba = pipeline.predict_proba(template)[:, 1][0]
        label = int(proba >= 0.5)

        st.markdown("---")
        st.markdown("### Prediction Result")
        st.metric("Churn Probability", f"{proba * 100:.1f}%")

        if label == 1:
            st.error("High churn risk. Consider proactive retention actions.")
        else:
            st.success("Low churn risk.")

# ---------- BATCH MODE ----------
elif mode == "Batch Prediction (CSV)":
    st.subheader("Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df_uploaded = pd.read_csv(file)
        st.write("Preview of uploaded data:")
        st.dataframe(df_uploaded.head())

        if df_example is None:
            st.error("Example dataset not found. Cannot align columns.")
            st.stop()

        if st.button("Run Batch Prediction"):
            # Align columns with training data
            template_cols = df_example.drop(columns=["Churn"]) if "Churn" in df_example.columns else df_example
            df_aligned = df_uploaded.reindex(columns=template_cols.columns, fill_value=np.nan)

            proba = pipeline.predict_proba(df_aligned)[:, 1]
            labels = (proba >= 0.5).astype(int)

            df_result = df_uploaded.copy()
            df_result["churn_probability"] = proba
            df_result["churn_pred"] = labels

            st.markdown("### Results")
            st.dataframe(df_result.head())

            csv = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download results as CSV",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )


# ---------- ABOUT / INFO ----------
else:
    st.subheader("About this app")
    st.write(
        """
        This app is built on the IBM Telco Customer Churn dataset.
        It uses a scikit-learn Pipeline (with preprocessing + Logistic Regression / Random Forest / XGBoost)
        trained offline and loaded here with Joblib.
        """
    )

    if df_example is not None:
        st.markdown("### Dataset Snapshot")
        st.dataframe(df_example.head())
    else:
        st.info("Example dataset not found at data/processed/telco_churn_clean.csv.")

    shap_path = os.path.join("reports", "figures", "shap_summary_plot.png")
    if os.path.exists(shap_path):
        st.markdown("### SHAP Summary Plot")
        st.image(shap_path, caption="Feature importance via SHAP", use_column_width=True)
    else:
        st.info("SHAP summary plot image not found.")