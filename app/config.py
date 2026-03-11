from pydantic import BaseModel

class Settings(BaseModel):
    artifacts_dir: str = "artifacts"
    model_file: str = "heart_rf.joblib"
    threshold: float = 0.5  # can tune later for screening (e.g., 0.4)

settings = Settings()