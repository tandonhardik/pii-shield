# trained_models.py
import os
from transformers import pipeline
import joblib

# Point this at either a local path or a Hugging Face Hub repo id, e.g.
# app/trained_models.py — change NER_MODEL_PATH default
# NER_MODEL_PATH = os.environ.get("NER_MODEL_PATH", "htandon7/pii-ner-distilbert")
NER_MODEL_PATH = os.environ.get("NER_MODEL_PATH", "../models/pii-ner-model-final")
SENSITIVITY_CLF_PATH = os.environ.get("SENSITIVITY_CLF_PATH", "../models/sensitivity_clf.pkl")
SENSITIVITY_VEC_PATH = os.environ.get("SENSITIVITY_VEC_PATH", "../models/sensitivity_vec.pkl")

your_ner = pipeline(
    "ner",
    model=NER_MODEL_PATH,
    tokenizer=NER_MODEL_PATH,
    aggregation_strategy="simple",
)

sens_clf = joblib.load(SENSITIVITY_CLF_PATH)
sens_vec = joblib.load(SENSITIVITY_VEC_PATH)


def detect_with_trained_model(text: str):
    raw = your_ner(text)
    return [
        {
            "entity_type": e["entity_group"],
            "start": e["start"],
            "end": e["end"],
            "text": e["word"],
            "score": float(e["score"]),
        }
        for e in raw
    ]


def classify_sensitivity(text: str) -> str:
    return sens_clf.predict(sens_vec.transform([text]))[0]
