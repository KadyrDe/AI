from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime

class PredictRequest(BaseModel):
    # Flexible payload: allows any fields (your dataset columns)
    features: Dict[str, Any] = Field(..., description="Patient features mapped to dataset columns")

class PredictResponse(BaseModel):
    possible_conditions: List[str]
    risk_level: str
    urgency: str
    recommended_specialists: List[str]
    recommended_tests: List[str]
    explanation: str
    confidence_level: float
    prediction: int
    model_version: Optional[str] = None
    threshold_used: float
    disclaimer: str
    timestamp: datetime