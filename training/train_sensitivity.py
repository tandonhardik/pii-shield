# train_sensitivity.py
#
# Trains a lightweight TF-IDF + Logistic Regression classifier for
# 3-class prompt sensitivity: PUBLIC / QUASI_IDENTIFIER / HIGH_RISK.
#
# KNOWN LIMITATION (see eval/results/sensitivity_ood.md): trained on 9
# fixed templates, so it hits 1.00 F1 on a held-out split of the SAME
# templates (template memorization) but only ~70% on hand-written,
# non-templated out-of-distribution sentences. Kept as-is for the
# prototype; documented rather than hidden. A fine-tuned transformer
# would likely close this gap -- see README roadmap.

from faker import Faker
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

fake = Faker()

templates = {
    "PUBLIC": [
        "Summarize this open-source paper about {topic}",
        "What is the capital of {country}",
        "{org} released a new product today",
        "Explain how {topic} works",
    ],
    "QUASI_IDENTIFIER": [
        "A {age}-year-old {gender} from {city} asked about {topic}",
        "The client works at {org} in {city}",
        "Someone born in {city} contacted us",
    ],
    "HIGH_RISK": [
        "Patient {name}, SSN {ssn}, has {condition}",
        "{name}'s phone number is {phone} and diagnosis is {condition}",
        "Contact {name} at {phone} regarding their {condition} treatment",
    ],
}


def fill(t):
    return t.format(
        topic=fake.word(), country=fake.country(), org=fake.company(),
        age=random.randint(18, 90), gender=random.choice(["male", "female"]),
        city=fake.city(), name=fake.name(), ssn=fake.ssn(),
        condition=random.choice(["diabetes", "asthma", "hypertension", "anxiety disorder"]),
        phone=fake.phone_number(),
    )


if __name__ == "__main__":
    data = []
    for label, temps in templates.items():
        for _ in range(300):
            data.append((fill(random.choice(temps)), label))

    texts, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_train_v = vec.fit_transform(X_train)
    X_test_v = vec.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_v, y_train)
    print("In-template held-out split report:")
    print(classification_report(y_test, clf.predict(X_test_v)))

    joblib.dump(clf, "../models/sensitivity_clf.pkl")
    joblib.dump(vec, "../models/sensitivity_vec.pkl")
    print("Saved to ../models/")
