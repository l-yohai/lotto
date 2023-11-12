import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import ordinal


def inference():
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)

    model_name_or_path = "l-yohai/bert_base_lotto"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=45, problem_type="multi_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=10,
    )

    target_draw_no = df.iloc[-1]["drwNo"] + 1
    text = f"{ordinal(target_draw_no)} lottery numbers"
    outputs = model(**tokenizer(text, return_tensors="pt")).logits
    preds = torch.topk(outputs, 7).indices
    print(sorted([p + 1 for p in preds[0].tolist()]))

    with open(f"predicted/{ordinal(target_draw_no)}_lottery_numbers.txt", "w") as f:
        f.write(
            f"{ordinal(target_draw_no)} predicted numbers: {', '.join(map(str, sorted([p + 1 for p in preds[0].tolist()])))}"
        )


if __name__ == "__main__":
    inference()
