from datetime import datetime
from .config import settings

def build_response(proba: float, pred: int) -> dict:

    if proba >= 0.75:
        risk = "high"
        urgency = "consult within 48 hours"
    elif proba >= 0.5:
        risk = "medium"
        urgency = "consult within 1-2 weeks"
    else:
        risk = "low"
        urgency = "routine consult if symptoms persist"

    return {
        "possible_conditions": [f"Heart disease likelihood: {round(proba*100)}%"],
        "risk_level": risk,
        "urgency": urgency,
        "recommended_specialists": ["Cardiologist"],
        "recommended_tests": ["ECG", "Lipid profile"],
        "explanation": "MVP model trained on UCI Heart Disease (binary: num>0). Not a diagnosis.",
        "confidence_level": round(proba, 3),
        "prediction": pred,
        "model_version": "rf_v1",
        "threshold_used": settings.threshold,
        "disclaimer": "This analysis is informational only and does not constitute a medical diagnosis. Please consult a licensed physician.",
        "timestamp": datetime.utcnow()
    }