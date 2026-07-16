# test_pipeline.py
#
# Basic pytest coverage for the behavior manually verified during
# development (see eval/results/ for the full benchmark writeups).
# Run from repo root with the app/ dir on PYTHONPATH, e.g.:
#   cd app && pytest ../tests/test_pipeline.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest
from privacy_accountant import SessionPrivacyAccountant
from reversible_anonymizer import ReversibleAnonymizer


def test_risk_accumulates_across_turns():
    acc = SessionPrivacyAccountant()
    r1 = acc.update_and_check("s1", [{"entity_type": "AGE", "start": 0, "end": 2, "text": "45"}])
    r2 = acc.update_and_check("s1", [{"entity_type": "ZIP5", "start": 0, "end": 5, "text": "60614"}])
    r3 = acc.update_and_check("s1", [{"entity_type": "MEDICAL_CONDITION", "start": 0, "end": 8, "text": "diabetes"}])
    assert r1 < r2 < r3
    assert r3 == pytest.approx(0.406, abs=0.01)


def test_ssn_dominates_risk_score():
    acc = SessionPrivacyAccountant()
    risk = acc.update_and_check(
        "s2", [{"entity_type": "US_SSN", "start": 0, "end": 11, "text": "123-45-6789"}]
    )
    assert risk > 0.9  # SSN weight (0.99) should dominate, not be near-zero


def test_unmapped_entity_type_does_not_crash():
    acc = SessionPrivacyAccountant()
    risk = acc.update_and_check("s3", [{"entity_type": "UNKNOWN_TYPE", "start": 0, "end": 3, "text": "xyz"}])
    assert risk == 0.0  # unmapped entities contribute nothing, don't error


def test_reversible_anonymizer_round_trips():
    ra = ReversibleAnonymizer()
    text = "My SSN is 123-45-6789"
    entities = [{"entity_type": "US_SSN", "start": 10, "end": 21, "text": "123-45-6789"}]
    sanitized, encrypted_map = ra.anonymize("s4", text, entities)
    assert "123-45-6789" not in sanitized
    assert "[US_SSN_" in sanitized

    fake_llm_response = sanitized  # pretend the LLM echoed it back unchanged
    restored = ra.deanonymize("s4", fake_llm_response, encrypted_map)
    assert "123-45-6789" in restored


def test_get_risk_returns_zero_for_unknown_session():
    acc = SessionPrivacyAccountant()
    assert acc.get_risk("never-seen-session") == 0.0
