# Lottery Number Prediction with Language Models

[![GitHub Repo stars](https://img.shields.io/github/stars/l-yohai/lotto?style=social)](https://github.com/l-yohai/lotto/stargazers)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub last commit](https://img.shields.io/github/last-commit/l-yohai/lotto)](https://github.com/l-yohai/lotto/commits/main)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/l-yohai/lotto/pulls)


## Project Description
This project focuses on training a multi-label classification model and sequence to sequence model using South Korean lottery number data. This project's goal is to predict future lottery numbers based on historical draws. I utilize Python, PyTorch, and the Hugging Face Transformers library for this purpose.

Disclaimer: This project is intended purely for entertainment purposes. Lottery draws are independent events, and the outcomes of previous draws have no bearing on future ones. This project should not be taken as a serious attempt to predict lottery numbers. Users are advised to view this as a reference and not to rely on it for gambling decisions.

***Additional Note***: Decisions to purchase lottery tickets based on this project's output are solely the responsibility of the viewer. The creator of this project bears no responsibility for any gambling decisions made based on the information provided here.

## Predicted Lottery Numbers

### Latest Prediction

- 1098th predicted numbers
    - bert: 11, 12, 18, 20, 21, 45, bonus: 22
    - bart: 1, 2, 10, 14, 20, 43, bonus: 13
- 1098th actual numbers:

### Previous Predictions

<details>
    <summary>1097th</summary>

- 1097th predicted numbers
    - bert: 2, 7, 10, 19, 33, 36, bonus: 22
    - bart: 2, 4, 12, 14, 18, 26, bonus: 35
- 1097th actual numbers: 14, 33, 34, 35, 37, 40, bonus: 4

</details>

<details>
    <summary>1096th</summary>

- 1096th predicted numbers
    - bert: 11, 19, 33, 36, 41, 45, bonus: 12
    - bart: 2, 4, 12, 14, 23, 34, bonus: 27
- 1096th actual numbers: 1, 12, 16, 19, 23, 43, bonus: 34

</details>

<details>
    <summary>1095th</summary>

- 1095th predicted numbers
    - bert: 11, 12, 18, 19, 24, 45, bonus: 21
    - bart: 4, 8, 17, 18, 22, 24, bonus: 27
- 1095th actual numbers: 8, 14, 28, 29, 34, 40, bonus: 12

</details>

<details>
    <summary>1094th</summary>

- 1094th predicted numbers
    - bert: 12, 18, 19, 33, 36, 42, bonus: 43
    - bart: 5, 12, 17, 18, 23, 26, bonus: 43
- 1094th actual numbers: 6, 7, 15, 22, 26, 40, bonus: 41

</details>

<details>
    <summary>1093rd</summary>

- 1093rd predicted numbers
    - bert: 6, 18, 22, 24, 35, 44, bonus: 45
    - bart: 4, 12, 14, 18, 30, 44, bonus: 21
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

## Citation

If this work is helpful, please kindly cite as:

```bibtex
@Misc{lotto,
  title = {Lottery Number Prediction with Language Models},
  author = {l-yohai},
  howpublished = {\url{https://github.com/l-yohai/lotto}},
  year = {2023}
}
```

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=l-yohai/lotto&type=Date)
