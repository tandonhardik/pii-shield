# NER Benchmark: Fine-tuned DistilBERT vs Presidio

## Setup

- Fine-tuned `distilbert-base-uncased` on 8,000 examples from
  `ai4privacy/pii-masking-200k` (3 epochs, T4 GPU, ~15-20 min).
- Evaluated on 54 held-out **English-only** examples (see note below on
  why the language filter matters).
- Compared against Presidio's out-of-the-box `AnalyzerEngine` (configured
  `language="en"`) on the same test set, restricted to entity types both
  systems attempt (excludes `ORG`, since the dataset's `JOBAREA`/`JOBTITLE`
  labels collapsed into `ORG` are not something Presidio's organization
  recognizer is designed to catch -- an unwinnable comparison for Presidio).

## ⚠️ Methodology note (found and fixed during development)

An earlier run of this benchmark used the dataset's full multilingual mix
(only ~21% of rows are English) without filtering. Presidio's `PERSON`
recognizer, run with `language="en"` against German/French/Italian text,
produced a large number of false positives that were a **language
configuration mismatch**, not a genuine detection failure -- e.g. it
tagged fragments of German banking-notification sentences as `PERSON`.
That run inflated the apparent gap between the two systems. The numbers
below are from a corrected, English-filtered re-run.

## Results (n=54 English test examples)

| Entity | Fine-tuned Model F1 | Presidio F1 | Notes |
|---|---|---|---|
| PERSON | 0.975 | 0.655 | Presidio over/under-fires on short templated context |
| LOCATION | 0.960 | 0.549 | |
| DATE | 0.973 | 0.500 | Presidio precision low (0.352) -- high false-positive rate |
| PHONE | 0.962 | 0.439 | |
| SSN | 0.922 | 0.209 | Dataset's numeric ID formats diverge from Presidio's narrow `US_SSN` regex |
| EMAIL | 0.971 | 1.000 | Both near-ceiling; email is largely regex-matchable |

## Honest framing

The fine-tuned model was trained on the dataset's full multilingual mix
(only ~21% English rows) yet generalizes well to English PII detection,
outperforming Presidio's out-of-the-box English recognizers on 5 of 6
comparable entity types. This is an **in-distribution comparison**: the
model was trained on this dataset's distribution; Presidio is a
general-purpose tool not tuned to it. The largest gaps are on SSN and
DATE, where Presidio's narrower built-in patterns underperform on this
dataset's format variety. Test set size (n=54) is modest -- results
should be read as indicative, not statistically definitive. Increasing
to n=500 (code in `eval/benchmark_ner.py`) would tighten confidence.
