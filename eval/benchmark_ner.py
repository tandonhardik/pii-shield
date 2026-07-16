# benchmark_ner.py
#
# Run this in the same Colab session right after training (needs `ds`,
# `your_ner`/model, and `analyzer` in scope). Produces per-entity-type
# precision/recall/F1 for your model vs Presidio, evaluated against the
# dataset's ground-truth `privacy_mask` spans -- NOT just raw entity
# counts, which is a confounded comparison (see README limitations for
# why: language mismatches and unwinnable entity-type comparisons were
# both found and corrected during development).
#
# IMPORTANT: filter to English-only rows before running this, since
# Presidio's analyzer is configured with language="en" -- running it on
# German/French/Italian rows produces spurious false positives that are
# a language-mismatch artifact, not a real detection-quality signal.

from collections import defaultdict


def get_ground_truth_spans(example, label_map):
    spans = []
    for item in example["privacy_mask"]:
        mapped = label_map.get(item["label"])
        if mapped:
            spans.append((item["start"], item["end"], mapped))
    return spans


def spans_overlap(a_start, a_end, b_start, b_end, threshold=0.5):
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    return overlap / max(a_end - a_start, 1) >= threshold


def score_model_per_type(get_predicted_spans_fn, test_examples, entity_types, label_map):
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for ex in test_examples:
        text = ex["source_text"]
        gt_spans = [s for s in get_ground_truth_spans(ex, label_map) if s[2] in entity_types]
        pred_spans = [s for s in get_predicted_spans_fn(text) if s[2] in entity_types]

        matched_gt = set()
        for p_start, p_end, p_type in pred_spans:
            found = False
            for i, (g_start, g_end, g_type) in enumerate(gt_spans):
                if i in matched_gt:
                    continue
                if p_type == g_type and spans_overlap(p_start, p_end, g_start, g_end):
                    matched_gt.add(i)
                    found = True
                    break
            if found:
                stats[p_type]["tp"] += 1
            else:
                stats[p_type]["fp"] += 1
        for i, (g_start, g_end, g_type) in enumerate(gt_spans):
            if i not in matched_gt:
                stats[g_type]["fn"] += 1

    results = {}
    for etype, s in stats.items():
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        results[etype] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), **s}
    return results


if __name__ == "__main__":
    # Example usage inside the Colab session (after training + loading Presidio):
    #
    # from presidio_analyzer import AnalyzerEngine
    # analyzer = AnalyzerEngine()
    #
    # def your_model_spans(text):
    #     ents = your_ner(text)
    #     return [(e["start"], e["end"], e["entity_group"]) for e in ents]
    #
    # PRESIDIO_MAP = {
    #     "PERSON": "PERSON", "LOCATION": "LOCATION", "DATE_TIME": "DATE",
    #     "PHONE_NUMBER": "PHONE", "US_SSN": "SSN", "ORGANIZATION": "ORG",
    #     "EMAIL_ADDRESS": "EMAIL",
    # }
    # def presidio_spans(text):
    #     results = analyzer.analyze(text=text, language="en")
    #     return [(r.start, r.end, PRESIDIO_MAP[r.entity_type]) for r in results if r.entity_type in PRESIDIO_MAP]
    #
    # FAIR_COMPARISON_TYPES = {"PERSON", "LOCATION", "DATE", "PHONE", "SSN", "EMAIL"}
    # test_examples = ds["train"].filter(lambda x: x["language"] == "en").shuffle(seed=99).select(range(200))
    #
    # print("Your model:", score_model_per_type(your_model_spans, test_examples, FAIR_COMPARISON_TYPES, LABEL_MAP))
    # print("Presidio:", score_model_per_type(presidio_spans, test_examples, FAIR_COMPARISON_TYPES, LABEL_MAP))
    print("See usage example in this file's __main__ block / training/train_ner_colab.md")
