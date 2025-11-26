import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path

from .config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    path = path or str(RAW_DATA_PATH)
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # Map target
    df[TARGET_COL] = df[TARGET_COL].map({'No': 0, 'Yes': 1}).astype(int)

    return df


def save_processed_data(df: pd.DataFrame, path: str | None = None) -> None:
    path = path or str(PROCESSED_DATA_PATH)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    return X, y


def get_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )

    return preprocessor, cat_cols, num_cols


def train_test_split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )