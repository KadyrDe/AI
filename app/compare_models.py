import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/heart_disease_uci.csv"
TARGET_COL = "num"

df = pd.read_csv(DATA_PATH)

if "id" in df.columns:
    df = df.drop(columns=["id"])

y = (df[TARGET_COL].astype(float) > 0).astype(int)
X = df.drop(columns=[TARGET_COL])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Preprocessing
numeric_pipe_lr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_lr = ColumnTransformer([
    ("num", numeric_pipe_lr, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

preprocess_rf = ColumnTransformer([
    ("num", numeric_pipe_rf, num_cols),
    ("cat", categorical_pipe, cat_cols)
])

lr = Pipeline([
    ("preprocess", preprocess_lr),
    ("model", LogisticRegression(max_iter=2000))
])

rf = Pipeline([
    ("preprocess", preprocess_rf),
    ("model", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf
}

for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\n=== {name} ===")
    print("Confusion Matrix:\n", cm)
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("Sensitivity (Recall+):", round(sensitivity, 3))
    print("Specificity (Recall-):", round(specificity, 3))
    print(classification_report(y_test, pred, digits=3))