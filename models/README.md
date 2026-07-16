# models/

Place your trained artifacts here (gitignored for large files):

- `pii-ner-model-final/` — unzip the fine-tuned DistilBERT NER model from Colab here
  (or better: publish to Hugging Face Hub and point `NER_MODEL_PATH` in
  `app/trained_models.py` at the Hub repo id instead)
- `sensitivity_clf.pkl` — output of `training/train_sensitivity.py`
- `sensitivity_vec.pkl` — output of `training/train_sensitivity.py`
