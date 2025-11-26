import pandas as pd
from typing import Dict

from .model_training import load_pipeline


def predict_single_customer(customer_dict: Dict, threshold: float = 0.5) -> Dict:
    pipeline = load_pipeline()

    customer_df = pd.DataFrame([customer_dict])
    prob = pipeline.predict_proba(customer_df)[:, 1][0]
    label = int(prob >= threshold)

    return {
        "churn_probability": float(prob),
        "churn_label": label
    }


def predict_from_csv(input_path: str, output_path: str, threshold: float = 0.5):
    pipeline = load_pipeline()

    df = pd.read_csv(input_path)
    probs = pipeline.predict_proba(df)[:, 1]
    labels = (probs >= threshold).astype(int)

    df['churn_probability'] = probs
    df['churn_label'] = labels

    df.to_csv(output_path, index=False)
    return output_path