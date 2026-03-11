import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

DATA_PATH = "data/heart_disease_uci.csv"
TARGET_COL = "num"
MODEL_PATH = "artifacts/model.joblib"

def eval_at_threshold(proba, y_true, thr: float):
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return cm, sensitivity, specificity, classification_report(y_true, pred, digits=3)

def main():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]

    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    y = (df[TARGET_COL].astype(float) > 0).astype(int)
    X = df.drop(columns=[TARGET_COL])

    # same split logic for consistent evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print("ROC-AUC:", auc)

    for thr in [0.5, 0.4]:
        cm, sens, spec, rep = eval_at_threshold(proba, y_test, thr)
        print(f"\n=== Threshold {thr} ===")
        print("Confusion matrix:\n", cm)
        print("Sensitivity (Recall+):", round(sens, 3))
        print("Specificity (Recall-):", round(spec, 3))
        print(rep)

if __name__ == "__main__":
    main()