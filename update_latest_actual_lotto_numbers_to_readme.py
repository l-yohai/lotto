import pandas as pd

from utils import ordinal


def update_readme_with_actual_numbers(
    readme_path,
    draw_no,
    actual_numbers_text,
):
    # README 파일 읽기
    with open(readme_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 새로운 실제 번호 내용 추가
    new_content = f"- {actual_numbers_text}\n"
    archived_content = ""
    updated = False

    for i, line in enumerate(lines):
        if line.startswith(f"- {ordinal(draw_no)} predicted numbers"):
            if not updated:
                # 현재 예측 내용을 저장하고, 실제 번호 업데이트
                archived_content = "".join(lines[i : i + 3])
                lines[i : i + 4] = ""
                updated = True
            else:
                # 'Latest Prediction' 초기화
                lines[i] = lines[i].split(":")[0] + "\n"
                lines[i + 1] = lines[i + 1].split(":")[0] + "\n"
        elif line.strip().startswith("### Previous Predictions") and archived_content:
            # 이전 예측 섹션에 현재 회차 내용 추가
            lines.insert(
                i + 1,
                f"\n<details>\n    <summary>{ordinal(draw_no)}</summary>\n\n{archived_content + new_content}\n</details>\n",
            )
            break

    # README 파일 다시 쓰기
    with open(readme_path, "w", encoding="utf-8") as file:
        file.writelines(lines)


if __name__ == "__main__":
    # 예시 사용
    readme_path = "README.md"
    df = pd.read_json("data/lotto_numbers.jsonl", lines=True)
    latest_info = df.iloc[-1]

    actual_numbers = [int(latest_info[f"drwtNo{i}"]) for i in range(1, 7)] + [
        latest_info["bnusNo"]
    ]
    draw_no = latest_info["drwNo"]

    actual_numbers_text = f"{ordinal(draw_no)} actual numbers: {', '.join(map(str, sorted([num for num in actual_numbers])))}"

    update_readme_with_actual_numbers(readme_path, draw_no, actual_numbers_text)
