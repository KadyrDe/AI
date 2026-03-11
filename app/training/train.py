import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/heart_disease_uci.csv"
TARGET_COL = "num"

# куда сохраняем артефакты (под app/config.py)
MODEL_OUT = "artifacts/model.joblib"
METADATA_OUT = "artifacts/metadata.json"

def main():
    df = pd.read_csv(DATA_PATH)

    # Drop ID column (not a medical feature)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Binary target: 0 -> no disease, 1..4 -> disease
    y_raw = df[TARGET_COL]
    y = (y_raw.astype(float) > 0).astype(int)

    X = df.drop(columns=[TARGET_COL])

    # Identify numeric vs categorical
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))

    bundle = {
        "pipeline": clf,
        "target_col": TARGET_COL,
        "feature_cols": X.columns.tolist()
    }
    joblib.dump(bundle, MODEL_OUT)

    metadata = {
        "model_version": "rf_v1",
        "dataset_path": DATA_PATH,
        "target_col": TARGET_COL,
        "n_rows": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "roc_auc_test": auc,
        "notes": "Binary target: num>0"
    }
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metadata: {METADATA_OUT}")
    print("Test ROC-AUC:", auc)

if __name__ == "__main__":
    main()