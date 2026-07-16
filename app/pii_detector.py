# pii_detector.py
from presidio_analyzer import AnalyzerEngine
import spacy
import re

<<<<<<< HEAD
nlp = spacy.load("en_core_web_lg")
_analyzer = AnalyzerEngine(supported_languages=["en"])
=======
<<<<<<< HEAD
nlp = spacy.load("en_core_web_sm")

# Explicitly tell Presidio to use the SAME small model, instead of
# silently downloading its own en_core_web_lg internally.
_nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}
_provider = NlpEngineProvider(nlp_configuration=_nlp_configuration)
_nlp_engine = _provider.create_engine()

_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])
=======
nlp = spacy.load("en_core_web_lg")
_analyzer = AnalyzerEngine(supported_languages=["en"])
>>>>>>> f7d7bf2 (Change)
>>>>>>> ca355de (Change)

# Entity types treated as catastrophic-if-missed -- used by main.py as a
# narrow deterministic safety net on top of the sensitivity classifier.
HIGH_RISK_TYPES = {"US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "IBAN"}


def contains_sensitive_pii(text: str) -> bool:
    """Quick check for catastrophic-if-wrong entity types only."""
    results = _analyzer.analyze(text=text, language="en")
    return any(res.entity_type in HIGH_RISK_TYPES for res in results)


def detect_pii_with_context(text: str):
    """
    Full PII detection: Presidio + spaCy DATE + regex-based MEDICAL_CONDITION,
    AGE, and ZIP5. Regex entities take priority over Presidio's broader
    DATE_TIME spans when they overlap, to avoid double/garbled hydration.
    """
    results = _analyzer.analyze(text=text, language="en")
    entities = []
    for res in results:
        entities.append({
            "entity_type": res.entity_type,
            "start": res.start,
            "end": res.end,
            "text": text[res.start:res.end],
            "score": res.score,
        })

    # Extra context from spaCy (DATE only -- spaCy has no native AGE label)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "DATE":
            if not any(e["start"] == ent.start_char and e["end"] == ent.end_char for e in entities):
                entities.append({
                    "entity_type": "DATE",
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "text": ent.text,
                    "score": 1.0,
                })

    # Medical conditions via regex (extend this list as needed)
    condition_pattern = r"\b(asthma|diabetes|hypertension|COPD|migraine|anxiety disorder)\b"
    for match in re.finditer(condition_pattern, text, re.IGNORECASE):
        if not any(e["start"] == match.start() and e["end"] == match.end() for e in entities):
            entities.append({
                "entity_type": "MEDICAL_CONDITION",
                "start": match.start(),
                "end": match.end(),
                "text": match.group(),
                "score": 1.0,
            })

    # AGE -- "45 years old", "45 y/o", "45 yo"
    age_pattern = r"\b(\d{1,3})\s*(?:years?[\s-]old|y/?o\b)"
    for match in re.finditer(age_pattern, text, re.IGNORECASE):
        entities.append({
            "entity_type": "AGE",
            "start": match.start(1),
            "end": match.end(1),
            "text": match.group(1),
            "score": 1.0,
        })

    # ZIP5 -- "zip code 60614"
    zip_pattern = r"\bzip\s*code\s*(\d{5})\b"
    for match in re.finditer(zip_pattern, text, re.IGNORECASE):
        entities.append({
            "entity_type": "ZIP5",
            "start": match.start(1),
            "end": match.end(1),
            "text": match.group(1),
            "score": 1.0,
        })

    # Drop any broad DATE/DATE_TIME entity that overlaps a more specific
    # AGE/ZIP5 span -- prevents double or conflicting hydration downstream.
    specific_spans = [(e["start"], e["end"]) for e in entities if e["entity_type"] in ("AGE", "ZIP5")]

    def _overlaps_specific(e):
        return any(not (e["end"] <= s or e["start"] >= en) for s, en in specific_spans)

    entities = [
        e for e in entities
        if not (e["entity_type"] in ("DATE", "DATE_TIME") and _overlaps_specific(e))
    ]

    return entities


def _spans_overlap(a, b, threshold: float = 0.5) -> bool:
    overlap = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    span_len = max(a["end"] - a["start"], 1)
    return overlap / span_len >= threshold


def detect_pii_ensemble(text: str):
    """
    Merge Presidio+regex detection with the fine-tuned NER model's output.
    Trained-model entities are only added when they don't overlap an
    existing Presidio/regex entity, to avoid duplicate/conflicting spans.
    """
    from trained_models import detect_with_trained_model  # local import avoids circular import at module load

    base_entities = detect_pii_with_context(text)
    trained_entities = detect_with_trained_model(text)

    merged = list(base_entities)
    for t_ent in trained_entities:
        if not any(_spans_overlap(t_ent, b_ent) for b_ent in base_entities):
            merged.append(t_ent)
    return merged
