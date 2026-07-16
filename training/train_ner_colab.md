# NER Model Training (Google Colab, T4 GPU)

This model was trained on Colab, not locally, since fine-tuning DistilBERT
on CPU is impractically slow. Full step-by-step notebook cells below --
copy into a new Colab notebook (Runtime -> Change runtime type -> T4 GPU).

## Cell 1 -- install + load dataset

```python
!pip install -q transformers datasets seqeval evaluate accelerate

from datasets import load_dataset
ds = load_dataset("ai4privacy/pii-masking-200k")
print(ds)
print(ds["train"][0])
```

## Cell 2 -- label collapse + target set

Collapses ai4privacy's ~50 fine-grained labels down to 7 target types:
PERSON, LOCATION, DATE, PHONE, SSN, ORG, EMAIL.

```python
LABEL_MAP = {
    "PREFIX": "PERSON", "FIRSTNAME": "PERSON", "LASTNAME": "PERSON",
    "MIDDLENAME": "PERSON", "USERNAME": "PERSON",
    "CITY": "LOCATION", "STATE": "LOCATION", "STREET": "LOCATION",
    "SECONDARYADDRESS": "LOCATION", "ZIPCODE": "LOCATION", "COUNTY": "LOCATION",
    "BUILDINGNUMBER": "LOCATION", "NEARBYGPSCOORDINATE": "LOCATION",
    "ORDINALDIRECTION": "LOCATION",
    "DATE": "DATE", "TIME": "DATE", "DOB": "DATE",
    "PHONENUMBER": "PHONE", "PHONEIMEI": "PHONE",
    "SSN": "SSN", "SOCIALNUMBER": "SSN", "IDCARD": "SSN", "PIN": "SSN",
    "PASSWORD": "SSN", "ACCOUNTNUMBER": "SSN", "CREDITCARDNUMBER": "SSN",
    "CREDITCARDCVV": "SSN", "IBAN": "SSN", "BIC": "SSN",
    "COMPANYNAME": "ORG", "EMPLOYER": "ORG", "JOBTITLE": "ORG",
    "JOBAREA": "ORG", "JOBTYPE": "ORG",
    "EMAIL": "EMAIL",
}
TARGET_ENTITIES = ["PERSON", "LOCATION", "DATE", "PHONE", "SSN", "ORG", "EMAIL"]
label_list = ["O"] + [f"{p}-{e}" for e in TARGET_ENTITIES for p in ("B", "I")]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

train_subset = ds["train"].shuffle(seed=42).select(range(8000))
val_subset = ds["train"].shuffle(seed=42).select(range(8000, 9000))
```

## Cell 3 -- tokenize with offset mapping, align from `privacy_mask` char spans

Uses the tokenizer's own `offset_mapping` for alignment (not whitespace
splitting), which handles punctuation/hyphenated values like IMEI numbers
correctly.

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
).to("cuda")

def tokenize_and_align(example):
    text = example["source_text"]
    encoding = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    spans = []
    for item in example["privacy_mask"]:
        mapped = LABEL_MAP.get(item["label"])
        if mapped:
            spans.append((item["start"], item["end"], mapped))

    labels = []
    for (tok_start, tok_end) in offsets:
        if tok_start == tok_end:
            labels.append(-100)
            continue
        assigned = "O"
        for span_start, span_end, ent_type in spans:
            if tok_start >= span_start and tok_end <= span_end:
                assigned = f"B-{ent_type}" if tok_start == span_start else f"I-{ent_type}"
                break
        labels.append(label2id[assigned])

    encoding["labels"] = labels
    encoding.pop("offset_mapping")
    return encoding

train_tok = train_subset.map(tokenize_and_align, remove_columns=train_subset.column_names)
val_tok = val_subset.map(tokenize_and_align, remove_columns=val_subset.column_names)
```

**Sanity check before training on the full set:**
```python
sample = train_subset[0]
result = tokenize_and_align(sample)
tokens = tokenizer.convert_ids_to_tokens(result["input_ids"])
for tok, lab in zip(tokens, result["labels"]):
    if lab != -100:
        print(tok, "->", id2label[lab])
```

## Cell 4 -- train

```python
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[label_list[l] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"]}

args = TrainingArguments(
    output_dir="./pii-ner-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
trainer.save_model("./pii-ner-model-final")
tokenizer.save_pretrained("./pii-ner-model-final")
```

## Cell 5 -- download and place into repo

```python
!zip -r pii-ner-model.zip pii-ner-model-final
from google.colab import files
files.download("pii-ner-model.zip")
```

Unzip into `models/pii-ner-model-final/` in this repo, OR (recommended)
publish to the Hugging Face Hub and point `NER_MODEL_PATH` in
`app/trained_models.py` at the Hub repo id instead of a local path --
avoids Git LFS / large-file issues entirely:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("yourusername/pii-ner-distilbert")
api.upload_folder(folder_path="./pii-ner-model-final", repo_id="yourusername/pii-ner-distilbert")
```

See `eval/benchmark_ner.py` for the evaluation harness used to produce
the numbers in `eval/results/ner_benchmark.md`.
