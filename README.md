# 🛡️ PII Shield

A context-aware PII detection and anonymization middleware for LLM prompts. Instead of flat token redaction (`[PERSON_1]`), it replaces sensitive data with realistic, privacy-safe proxies that preserve sentence structure and semantic relationships — while tracking cumulative re-identification risk across multi-turn conversations.

[Live demo](#) · [Screenshot/GIF here]

## What it does

- **Context-aware PII detection** — ensemble of Presidio's rule/model-based recognizers + a fine-tuned DistilBERT NER model, plus regex recognizers for medical conditions, age, and ZIP codes.
- **Semantic Proxy Hydration** — sensitive values are swapped for realistic-but-fake equivalents (a name becomes another name, a city becomes a similarly-sized city, a disease becomes a related condition) instead of a generic placeholder — preserving downstream reasoning quality.
- **Two anonymization modes**:
  - *Hydration* (irreversible) — the LLM sees plausible fake values; nothing is recoverable afterward. Good when you don't need the exact original back.
  - *Reversible* — the LLM sees opaque encrypted tokens; the real values are restored client-side after the response. Good for clinical/legal use cases needing exact data.
- **Intent-driven policy routing** — a lightweight trained classifier judges whether a prompt is genuinely public and can bypass anonymization entirely (near-zero latency), layered with a narrow deterministic safety net for entity types where a false negative would be a serious leak (SSN, credit card).
- **Stateful multi-turn privacy accountant** — tracks disclosed quasi-identifiers (age, ZIP, medical condition, etc.) across an entire session and computes cumulative re-identification risk, so it can catch identity reconstruction from fragments trickled across many prompts, not just single-prompt leaks.

## Architecture

```
prompt
  │
  ▼
[Sensitivity classifier] ──PUBLIC & no catastrophic entities──► bypass, send as-is (zero overhead)
  │
  ▼ (otherwise)
[PII detection ensemble: Presidio + fine-tuned NER + regex]
  │
  ▼
[Semantic Proxy Hydration  OR  Reversible token mapping (encrypted)]
  │
  ▼
[Session Privacy Accountant: update cumulative risk]
  │
  ▼
sanitized prompt → LLM → (deanonymize if reversible mode) → response
```

## Benchmarks

### PII detection: fine-tuned DistilBERT vs Presidio

Evaluated on 54 held-out English examples from `ai4privacy/pii-masking-200k`, restricted to entity types Presidio has comparable recognizers for.

| Entity | Fine-tuned Model F1 | Presidio F1 |
|---|---|---|
| PERSON | 0.975 | 0.655 |
| LOCATION | 0.960 | 0.549 |
| DATE | 0.973 | 0.500 |
| PHONE | 0.962 | 0.439 |
| SSN | 0.922 | 0.209 |
| EMAIL | 0.971 | 1.000 |

This is an **in-distribution** comparison — the model was fine-tuned on this dataset's distribution, Presidio wasn't. Full methodology, an earlier flawed run that mixed languages (and why that inflated the gap), and per-entity error analysis: [`eval/results/ner_benchmark.md`](eval/results/ner_benchmark.md).

### Sensitivity classifier: in-template vs out-of-distribution

| Eval | Accuracy/F1 |
|---|---|
| In-template held-out split | 1.00 F1 (template memorization — not a meaningful number alone) |
| Out-of-distribution (hand-written sentences) | 70% accuracy |

Full failure-mode analysis: [`eval/results/sensitivity_ood.md`](eval/results/sensitivity_ood.md).

## Known limitations

- **Sensitivity classifier generalization**: TF-IDF+LR trained on only 9 templates hits 70% on out-of-distribution phrasing, with a specific, diagnosed failure mode (over-associates relationship words with risk tier; misses non-lexical contextual quasi-identifiers). Documented in detail rather than hidden — see `eval/results/sensitivity_ood.md`. Planned fix: fine-tuned transformer classifier, same recipe as the NER model.
- **Risk scoring model**: the privacy accountant uses a simplified independence-assumption product rule over hand-tuned weights, not empirical population-uniqueness statistics (e.g. HIPAA Safe Harbor / k-anonymity literature). Good enough to demonstrate the concept of cumulative multi-turn risk; not a rigorous re-identification risk estimate.
- **NER training data is majority non-English** (~21% English rows) despite being evaluated and primarily used for English text — it generalizes well in practice, but this wasn't the original design intent.
- **In-memory session state**: both the privacy accountant and the reversible-anonymizer token map live in process memory and are lost on restart. Redis/SQLite persistence would be needed for production use.
- **Mock LLM**: swap `mock_llm()` in `app/main.py` for a real Anthropic/OpenAI client call before using this beyond a demo.
- **Single shared encryption key**: the reversible anonymizer uses one Fernet key for all sessions — production would need per-user/per-session keys.

## Setup

```bash
git clone <this-repo>
cd pii-shield
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Train the sensitivity classifier (fast, no GPU needed):
```bash
cd training && python train_sensitivity.py
```

Train the NER model on Colab (GPU) — see [`training/train_ner_colab.md`](training/train_ner_colab.md) for the full notebook, then unzip the result into `models/pii-ner-model-final/`.

Run the API:
```bash
cd app && uvicorn main:app --reload
```

Run the demo UI:
```bash
cd demo && streamlit run streamlit_app.py
```

Run tests:
```bash
pytest tests/
```

## Example

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","prompt":"My SSN is 123-45-6789 and I live in Detroit","policy":"auto","use_reversible":true}'
```

## Tech stack

FastAPI · Presidio · spaCy · Hugging Face Transformers (fine-tuned DistilBERT) · scikit-learn · Faker · cryptography (Fernet) · Streamlit

## Roadmap

- [ ] Fine-tuned transformer sensitivity classifier (replace TF-IDF+LR)
- [ ] Redis-backed session persistence
- [ ] Reversible multi-modal anonymization (text ↔ image/DICOM token alignment)
- [ ] Real population-statistics-based risk weights
