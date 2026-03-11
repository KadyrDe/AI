import joblib
from pathlib import Path
from .config import settings

_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        path = Path(settings.artifacts_dir) / settings.model_file
        _bundle = joblib.load(path)
    return _bundle