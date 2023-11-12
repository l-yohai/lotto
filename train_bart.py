import copy

import evaluate
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from utils import ordinal


def train():
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)

    data = []
    for i, row in df.iterrows():
        text = f"{ordinal(row['drwNo'])} lottery numbers"
        labels = [row[f"drwtNo{i}"] for i in range(1, 7)]
        label_texts = ""

        for i, label in enumerate(labels):
            label_texts += f"{ordinal(i + 1)} number is <num>{label}</num>\n"
        label_texts += f"Bonus number is <num>{row['bnusNo']}</num>"

        data.append({"text": text, "labels": label_texts})

    dataset = Dataset.from_pandas(pd.DataFrame(data))

    model_name_or_path = "facebook/bart-base"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=50,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<num>", "</num>"]})

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
    )
    model.resize_token_embeddings(len(tokenizer))

    def preprocess(example):
        texts = example["text"]
        inputs = tokenizer(
            texts,
        )
        labels = example["labels"]
        tokenized_labels = tokenizer(
            labels,
        )
        inputs["labels"] = tokenized_labels.input_ids
        return inputs

    dataset = dataset.map(
        preprocess,
        batched=True,
        num_proc=4,
        remove_columns=dataset.features,
        load_from_cache_file=False,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(pred):
        outputs = pred.predictions
        outputs[outputs == -100] = tokenizer.pad_token_id
        labels = pred.label_ids
        labels[labels == -100] = tokenizer.pad_token_id

        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=predictions, references=labels, tokenizer=tokenizer.tokenize
        )

        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir="l-yohai/bart_base_lotto",
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
        predict_with_generate=True,
        generation_max_length=80,
        fp16=True,
        push_to_hub=True,
        hub_strategy="end",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=copy.deepcopy(dataset).shuffle().select(range(100)),
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
