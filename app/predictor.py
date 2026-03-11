import pandas as pd
from .model_loader import load_bundle
from .config import settings

def predict_proba(features: dict) -> float:
    bundle = load_bundle()
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    X = pd.DataFrame([features])

    # Add missing columns as None
    for col in feature_cols:
        if col not in X.columns:
            X[col] = None

    X = X[feature_cols]
    proba = float(pipeline.predict_proba(X)[0, 1])
    return proba

def predict_label(proba: float) -> int:
    return int(proba >= settings.threshold)