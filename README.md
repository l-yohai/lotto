# Lottery Number Prediction with Language Models

## Project Description
This project focuses on training a multi-label classification model and sequence to sequence model using South Korean lottery number data. This project's goal is to predict future lottery numbers based on historical draws. I utilize Python, PyTorch, and the Hugging Face Transformers library for this purpose.

Disclaimer: This project is intended purely for entertainment purposes. Lottery draws are independent events, and the outcomes of previous draws have no bearing on future ones. This project should not be taken as a serious attempt to predict lottery numbers. Users are advised to view this as a reference and not to rely on it for gambling decisions.

***Additional Note***: Decisions to purchase lottery tickets based on this project's output are solely the responsibility of the viewer. The creator of this project bears no responsibility for any gambling decisions made based on the information provided here.

## Predicted Lottery Numbers

### Latest Prediction


### Previous Predictions

<details>
    <summary>1093rd</summary>

- 1093rd predicted numbers
    - bert: 6, 18, 22, 24, 35, 44, 45
    - bart: 4, 12, 14, 18, 21, 30, 44
- 1093rd actual numbers: 10, 17, 22, 30, 35, 43, 44

</details>


## Model Checkpoints

- [l-yohai/bert_base_lotto](https://huggingface.co/l-yohai/bert_base_lotto?text=1093rd+lottery+numbers)
- [l-yohai/bart_base_lotto](https://huggingface.co/l-yohai/bart_base_lotto?text=1093rd+lottery+numbers)


## Installation

Before using the project, you need to install the required libraries.

```bash
conda create -n lotto python=3.11.5
conda activate lotto

pip install -r requirements.txt
```

## Usage

1. Data Preparation: Prepare your lottery number data in the data/lotto_numbers.jsonl file.
    * using data/archive_lotto_numbers.py to crawl data from the official website.
2. Training: Run python train_bert.py or train_bart.py to train the model.
3. Evaluation: Execute python inference.py to assess the model's performance.

## License

This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.ko) license.
