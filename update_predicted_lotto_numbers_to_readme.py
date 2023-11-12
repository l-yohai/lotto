import pandas as pd

from utils import ordinal


def update_readme_with_prediction(
    readme_path,
    draw_no,
    predicted_numbers_text,
):
    # README 파일 읽기
    with open(readme_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 새로운 예측 내용 추가
    new_content = f"- {predicted_numbers_text}\n- {ordinal(draw_no)} actual numbers:\n"

    # 예측 번호 추가
    for i, line in enumerate(lines):
        if line.strip() == "### Latest Prediction":
            lines.insert(i + 2, new_content)
            break

    # README 파일 다시 쓰기
    with open(readme_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


if __name__ == "__main__":
    # 예시 사용
    readme_path = "README.md"
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)
    draw_no = df.iloc[-1]["drwNo"] + 1  # 회차 번호

    with open(f"predicted/{ordinal(draw_no)}_lottery_numbers.txt", "r") as f:
        predicted_numbers_text = f.read().strip()

    update_readme_with_prediction(readme_path, draw_no, predicted_numbers_text)
