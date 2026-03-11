from fastapi import FastAPI
from .schemas import PredictRequest, PredictResponse
from .predictor import predict_proba, predict_label
from .postprocess import build_response

app = FastAPI(title="MedAI AI Service", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    proba = predict_proba(req.features)
    pred = predict_label(proba)
    return build_response(proba, pred)