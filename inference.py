import re

import pandas as pd
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils import ordinal


def bert_inference(target_draw_no):
    model_name_or_path = "l-yohai/bert_base_lotto"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=45, problem_type="multi_label_classification"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=10,
    )

    text = f"{ordinal(target_draw_no)} lottery numbers"
    with torch.no_grad():
        outputs = model(**tokenizer(text, return_tensors="pt")).logits
    preds = torch.topk(outputs, 7).indices[0].tolist()
    nums, bonus = preds[:-1], preds[-1]

    result = sorted([p + 1 for p in nums]) + [f"bonus: {bonus + 1}"]
    print(f"bert inference: {result}")

    return result


def bart_inference(target_draw_no):
    model_name_or_path = "l-yohai/bart_base_lotto"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=50,
    )

    text = f"{ordinal(target_draw_no)} lottery numbers"

    extracted_numbers = []
    while len(list(set(extracted_numbers))) != 7:
        with torch.no_grad():
            outputs = model.generate(
                **tokenizer(text, return_tensors="pt"),
                max_new_tokens=80,
                do_sample=True,
                top_p=0.95,
            )
        decoded_text = tokenizer.batch_decode(outputs)[0]

        # "<num>" 태그 내의 숫자와 "number is" 다음의 숫자를 모두 포함하여 추출
        numbers = re.findall(r"<num>(\d+)</num>|(\d+)</num>", decoded_text)
        # 추출된 튜플에서 숫자만 추출하고 int로 변환
        extracted_numbers = [int(num) for nums in numbers for num in nums if num]

    nums, bonus = extracted_numbers[:-1], extracted_numbers[-1]
    result = sorted(nums) + [f"bonus: {bonus}"]
    print(f"bart inference: {result}")

    return result


def inference():
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)
    target_draw_no = df.iloc[-1]["drwNo"] + 1

    bert_inference_result = bert_inference(target_draw_no)
    bart_inference_result = bart_inference(target_draw_no)

    with open(f"predicted/{ordinal(target_draw_no)}_lottery_numbers.txt", "w") as f:
        f.write(f"- {ordinal(target_draw_no)} predicted numbers\n")
        f.write(f"    - bert: {', '.join(map(str, bert_inference_result))}\n")
        f.write(f"    - bart: {', '.join(map(str, bart_inference_result))}\n")


if __name__ == "__main__":
    inference()
