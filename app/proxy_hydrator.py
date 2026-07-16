# proxy_hydrator.py
from faker import Faker
import random

fake = Faker()
fake.seed_instance(42)

# Domain-specific swap tables -- extend these from public ontologies
# (e.g. ICD-10 chapters for diseases) for more realistic proxies.
CITY_SWAP = {
    "Chicago": "Detroit",
    "New York": "Philadelphia",
    "Los Angeles": "San Diego",
    "Houston": "Dallas",
}

DISEASE_SWAP = {
    "asthma": "COPD",
    "diabetes": "hypertension",
    "hypertension": "type 2 diabetes",
    "copd": "asthma",
    "migraine": "tension headache",
    "anxiety disorder": "panic disorder",
}


def get_proxy(entity_type: str, original_text: str, context=None) -> str:
    """Return a context-preserving fake replacement for a given entity type."""
    if entity_type == "PERSON":
        return fake.first_name() + " " + fake.last_name()
    elif entity_type in ("LOCATION", "GPE"):
        return CITY_SWAP.get(original_text, fake.city())
    elif entity_type in ("DATE_TIME", "DATE"):
        return str(fake.date())
    elif entity_type in ("US_SSN", "SSN"):
        return fake.ssn()
    elif entity_type == "CREDIT_CARD":
        return fake.credit_card_number()
    elif entity_type == "MEDICAL_CONDITION":
        return DISEASE_SWAP.get(original_text.lower(), "a common condition")
    elif entity_type == "AGE":
        try:
            age = int(original_text)
        except ValueError:
            age = 45
        return str(max(age + random.randint(-5, 5), 1))
    elif entity_type == "ZIP5":
        return str(random.randint(10000, 99999))
    elif entity_type in ("PHONE_NUMBER", "PHONE"):
        return fake.phone_number()
    elif entity_type in ("EMAIL_ADDRESS", "EMAIL"):
        return fake.email()
    else:
        return fake.word()


def semantic_hydrate(text: str, entities: list) -> str:
    """Replace all detected entities with semantic proxies (irreversible)."""
    sorted_ents = sorted(entities, key=lambda x: x["start"], reverse=True)
    for ent in sorted_ents:
        proxy = get_proxy(ent["entity_type"], ent["text"])
        text = text[:ent["start"]] + proxy + text[ent["end"]:]
    return text
