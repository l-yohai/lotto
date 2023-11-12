import copy

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from utils import ordinal


def compute_metrics(pred):
    labels = pred.label_ids
    outputs = pred.predictions

    metrics = {}

    top_indices = np.argsort(outputs, axis=1)[:, -6:]
    predictions = np.zeros_like(outputs)
    for i in range(outputs.shape[0]):
        predictions[i, top_indices[i]] = 1.0

    accs = []
    for label, top_index in zip(labels, top_indices):
        acc = len(set(np.where(label == 1)[0]) & set(sorted(top_index))) / 6
        accs.append(acc)
    accuracy = np.mean(accs)

    precision = precision_score(labels, predictions, average="micro")
    recall = recall_score(labels, predictions, average="micro")
    f1 = f1_score(labels, predictions, average="micro")

    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1

    return metrics


def encode_labels_as_vector(labels, max_number=45):
    """
    Encode a list of labels as a binary vector.
    :param labels: List of labels (e.g., lottery numbers)
    :param max_number: Maximum number (e.g., 45 for lottery)
    :return: Binary vector representing the labels
    """
    vector = [0.0] * max_number
    for label in labels:
        if 1.0 <= label <= float(max_number):
            vector[int(label) - 1] = 1.0
    return vector


def train():
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)

    data = []

    for _, row in df.iterrows():
        text = f"{ordinal(row['drwNo'])} lottery numbers"
        labels = [float(row[f"drwtNo{i}"]) for i in range(1, 7)]
        data.append({"text": text, "labels": labels})

    dataset = Dataset.from_pandas(pd.DataFrame(data))

    model_name_or_path = "bert-base-cased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=45, problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=10,
    )

    def preprocess(example):
        preprocessed_examples = tokenizer(
            example["text"],
            padding="max_length",
            max_length=10,
            return_tensors="pt",
        )
        labels = [encode_labels_as_vector(label) for label in example["labels"]]
        return {
            "input_ids": preprocessed_examples.input_ids,
            "attention_mask": preprocessed_examples.attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=dataset.features,
        load_from_cache_file=False,
    ).with_format("torch")

    training_args = TrainingArguments(
        output_dir="l-yohai/bert_base_lotto",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        num_train_epochs=10,
        learning_rate=5e-5,
        report_to="none",
        lr_scheduler_type="constant",
        fp16=True,
        push_to_hub=True,
        hub_strategy="end",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=copy.deepcopy(dataset).shuffle().select(range(100)),
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
