import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_churn_clean.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline_xgb.pkl"

TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42