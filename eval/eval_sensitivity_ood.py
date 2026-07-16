# eval_sensitivity_ood.py
#
# Tests the sensitivity classifier on hand-written sentences that do NOT
# come from its training templates. This is the number that actually
# matters -- the in-template train/test split scores 1.00 F1, which is
# template memorization, not generalization (see README limitations).

import joblib

clf = joblib.load("../models/sensitivity_clf.pkl")
vec = joblib.load("../models/sensitivity_vec.pkl")

ood_examples = [
    # PUBLIC
    ("Can you help me summarize the plot of Dune?", "PUBLIC"),
    ("What's a good recipe for banana bread?", "PUBLIC"),
    ("What's the weather like in Chicago this weekend?", "PUBLIC"),
    ("Explain how photosynthesis works to a 5th grader", "PUBLIC"),
    ("What are the best programming languages to learn in 2026?", "PUBLIC"),
    ("Summarize the main themes of this research paper on climate policy", "PUBLIC"),
    ("Give me some tips for staying productive while working from home", "PUBLIC"),
    ("What's the difference between a stock and a bond?", "PUBLIC"),
    ("Recommend a few good sci-fi books", "PUBLIC"),
    ("How do I fix a merge conflict in git?", "PUBLIC"),
    # QUASI_IDENTIFIER
    ("I'm a 34-year-old software engineer based in Austin", "QUASI_IDENTIFIER"),
    ("I work in marketing at a mid-sized firm downtown", "QUASI_IDENTIFIER"),
    ("My coworker mentioned she recently moved to Denver for a new job", "QUASI_IDENTIFIER"),
    ("A friend of mine, mid-40s, just started a small business in Ohio", "QUASI_IDENTIFIER"),
    ("She's originally from Seattle but has been living abroad for a few years", "QUASI_IDENTIFIER"),
    ("My neighbor is a retired teacher in his late 60s", "QUASI_IDENTIFIER"),
    ("I recently changed careers from finance to teaching", "QUASI_IDENTIFIER"),
    ("He's a grad student studying chemistry at a university in Boston", "QUASI_IDENTIFIER"),
    ("My sister just relocated to Miami for grad school", "QUASI_IDENTIFIER"),
    ("I'm in my late 20s and just moved to a new city for work", "QUASI_IDENTIFIER"),
    # HIGH_RISK
    ("My mom was just diagnosed with stage 2 breast cancer, her name is Linda Chen", "HIGH_RISK"),
    ("Can you draft an email to my doctor about my anxiety medication dosage, my number is 555-201-9988", "HIGH_RISK"),
    ("Here's my routing number and account number for the wire transfer: 021000021, 4457891023", "HIGH_RISK"),
    ("My social security number is 512-34-8890, can you help me fill out this form", "HIGH_RISK"),
    ("Patient John Meyer's blood test shows elevated glucose levels", "HIGH_RISK"),
    ("Please send the invoice to my home address, 214 Willow Creek Dr, and card ending 4471", "HIGH_RISK"),
    ("My therapist diagnosed me with generalized anxiety disorder last month", "HIGH_RISK"),
    ("I need help writing a letter disputing a charge on my credit card ending in 8823", "HIGH_RISK"),
    ("My son's pediatrician prescribed him medication for ADHD", "HIGH_RISK"),
    ("Can you help me draft a message to HR about my recent hospitalization", "HIGH_RISK"),
]

if __name__ == "__main__":
    texts, true_labels = zip(*ood_examples)
    preds = clf.predict(vec.transform(texts))

    correct = 0
    misses = []
    for text, true, pred in zip(texts, true_labels, preds):
        match = "PASS" if true == pred else "FAIL"
        if true == pred:
            correct += 1
        else:
            misses.append((text, true, pred))
        print(f"[{match}]  true={true:20s} pred={pred:20s} | {text}")

    print(f"\nOOD accuracy: {correct}/{len(ood_examples)} = {correct/len(ood_examples):.2%}")
    print("\nMisses:")
    for text, true, pred in misses:
        print(f"  [{true} -> {pred}] {text}")
