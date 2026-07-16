# Sensitivity Classifier: In-Template vs Out-of-Distribution Evaluation

## Setup

TF-IDF + Logistic Regression, 3-class (`PUBLIC` / `QUASI_IDENTIFIER` /
`HIGH_RISK`), trained on 900 synthetic examples generated from 9 fixed
sentence templates (Faker-filled).

## Result 1: In-template held-out split

```
                  precision    recall  f1-score   support
       HIGH_RISK       1.00      1.00      1.00        60
          PUBLIC       1.00      1.00      1.00        60
QUASI_IDENTIFIER       1.00      1.00      1.00        60
        accuracy                           1.00       180
```

**This number is not meaningful on its own.** Train and test both draw
from the same 9 templates -- the classifier can trivially separate
classes on fixed scaffolding words (e.g. "SSN", "diagnosis", "asked
about") without learning to generalize to new phrasing.

## Result 2: Out-of-distribution evaluation (hand-written, non-templated)

Run: `python eval_sensitivity_ood.py`

```
OOD accuracy: 21/30 = 70.00%
```

Failure modes identified:

- **PUBLIC -> QUASI_IDENTIFIER**: generic advice-seeking phrasing
  ("tips for staying productive", "best languages to learn") gets
  misclassified, likely because QUASI_IDENTIFIER templates all use
  personal-context openers ("I'm a...", "My coworker...").
- **QUASI_IDENTIFIER -> HIGH_RISK**: sentences mentioning a family
  relationship + location/age ("neighbor," "sister," "originally
  from") over-trigger HIGH_RISK, likely because HIGH_RISK templates
  also contain relationship words ("Patient," "son's," "mom"), so the
  model partly conflates "mentions a relationship" with "high risk"
  rather than detecting the actual medical/financial identifier.
- One clearly spurious case: "Recommend a few good sci-fi books" ->
  predicted HIGH_RISK with decision score +0.64, despite containing no
  PII-adjacent language at all -- consistent with a specific word
  picking up spurious weight from chance co-occurrence in the small
  (900-example, 9-template) training set.

## Conclusion

Training data diversity, not model capacity, is the primary bottleneck.
Next step: either (a) expand the template set substantially, or (b)
replace TF-IDF+LR with a fine-tuned transformer classifier (same recipe
as the NER model) to capture semantic context rather than lexical
co-occurrence -- (b) is the recommended fix, tracked in the README roadmap.
