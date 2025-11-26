from pathlib import Path

import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from .data_preprocessing import (
    load_raw_data,
    clean_data,
    save_processed_data,
    get_features_and_target,
    get_preprocessor,
    train_test_split_data,
)
from .config import MODEL_DIR, MODEL_PATH


def build_model():
    xgb_clf = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    return xgb_clf


def train_pipeline():
    # Load + clean
    df_raw = load_raw_data()
    df = clean_data(df_raw)
    save_processed_data(df)

    # Features / target
    X, y = get_features_and_target(df)

    # Preprocessor
    preprocessor, cat_cols, num_cols = get_preprocessor(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Pipeline
    model = build_model()
    pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    return pipeline, {"accuracy": accuracy, "roc_auc": roc_auc}


def save_pipeline(pipeline):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)


def load_pipeline():
    return joblib.load(MODEL_PATH)