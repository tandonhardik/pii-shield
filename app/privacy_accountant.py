# privacy_accountant.py

# Weights represent an approximate P(this attribute alone narrows the
# subject down to ~1 person). Near-unique identifiers (SSN, credit card)
# are close to 1; broad quasi-identifiers (ZIP, age) are small population
# fractions. NOTE: this is a heuristic simplification, not derived from
# empirical population-uniqueness statistics -- see README limitations.
QUASI_WEIGHTS = {
    "SSN": 0.99,
    "CREDIT_CARD": 0.99,
    "EMAIL": 0.9,
    "PHONE": 0.85,
    "FULL_NAME": 0.3,          # common names are far from unique alone
    "DISEASE_RARE": 0.4,       # rare condition + other context narrows significantly
    "ZIP5": 1 / 10_000,
    "ZIP9": 1 / 1_000_000,
    "AGE": 1 / 100,
    "WORKPLACE": 1 / 1_000,
    "CITY": 1 / 10_000,
}


class SessionPrivacyAccountant:
    def __init__(self):
        self.sessions = {}   # session_id -> {"disclosed": set(), "risk": float}

    def _map_entity(self, entity_type, value):
        if entity_type == "LOCATION" and value.isdigit() and len(value) == 5:
            return "ZIP5"
        elif entity_type == "ZIP5":
            return "ZIP5"
        elif entity_type == "MEDICAL_CONDITION":
            return "DISEASE_RARE"
        elif entity_type == "AGE":
            return "AGE"
        elif entity_type == "PERSON":
            return "FULL_NAME"
        elif entity_type in ("GPE", "LOCATION"):
            return "CITY"
        elif entity_type in ("US_SSN", "SSN"):
            return "SSN"
        elif entity_type in ("PHONE_NUMBER", "PHONE"):
            return "PHONE"
        elif entity_type == "CREDIT_CARD":
            return "CREDIT_CARD"
        elif entity_type in ("EMAIL_ADDRESS", "EMAIL"):
            return "EMAIL"
        return None

    def update_and_check(self, session_id, entities):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"disclosed": set(), "risk": 0.0}

        session = self.sessions[session_id]
        for ent in entities:
            qtype = self._map_entity(ent["entity_type"], ent["text"])
            if qtype:
                session["disclosed"].add(qtype)

        # Product rule: 1 - Pi(1 - weight) across all disclosed quasi-identifier types
        prob_not = 1.0
        for qtype in session["disclosed"]:
            weight = QUASI_WEIGHTS.get(qtype, 0.01)
            prob_not *= (1 - weight)
        risk = 1 - prob_not
        session["risk"] = risk
        return risk

    def get_risk(self, session_id):
        return self.sessions.get(session_id, {}).get("risk", 0.0)
