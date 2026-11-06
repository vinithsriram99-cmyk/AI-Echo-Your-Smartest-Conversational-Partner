"""Load the saved model and classify new text."""
import joblib

from config import MODEL_PATH
from preprocess import clean_text

_model = None


def predict_sentiment(text: str) -> dict:
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    cleaned = clean_text(text)
    proba = _model.predict_proba([cleaned])[0]
    probs = {c: round(float(p), 3) for c, p in zip(_model.classes_, proba)}
    return {"label": max(probs, key=probs.get), "probabilities": probs}


if __name__ == "__main__":
    for t in ["Terrible app, it keeps crashing", "It is okay, nothing special", "Excellent and very helpful!"]:
        print(t, "->", predict_sentiment(t))
