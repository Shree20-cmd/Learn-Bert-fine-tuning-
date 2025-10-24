import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, pipeline, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import classification_report

#  1. Load WikiNER (WORKS EVERYWHERE)
dataset = load_dataset("wikiann", "bn")
print(f" Loaded {len(dataset['train'])} training examples")
print(f"Labels: {dataset['train'].features['ner_tags'].feature.names}")

# 2. Model and Tokenizer
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

label_list = dataset["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, id2label=id2label, label2id=label2id, num_labels=num_labels
)

# 3. Tokenization and Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)

# 4. Train (smaller batch for faster testing)
training_args = TrainingArguments(
    output_dir="./wiki-ner-finetuned",
    eval_strategy="epoch",  # Changed from evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
)

print(" Training WikiNER...")
trainer.train()
trainer.save_model("./wiki-ner-finetuned")

# Test
ner_pipeline = pipeline("ner", model="./wiki-ner-finetuned", tokenizer=tokenizer, aggregation_strategy="simple")
print(ner_pipeline("Elon Musk founded Tesla in California"))