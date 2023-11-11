import requests
import json
import os

def fetch_latest_draw_no(jsonl_file):
    last_draw_no = 0
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as file:
            for line in file:
                lotto_data = json.loads(line.strip())
                last_draw_no = max(last_draw_no, lotto_data.get("drwNo", 0))
    return last_draw_no

def fetch_lotto_numbers(draw_no):
    url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}'
    response = requests.get(url)
    return response.json() if response.status_code == 200 else None

def append_to_jsonl(jsonl_file, data):
    with open(jsonl_file, 'a') as file:
        json_record = json.dumps(data)
        file.write(json_record + '\n')

def main():
    jsonl_file = '/mnt/data/lotto_numbers.jsonl'
    last_draw_no = fetch_latest_draw_no(jsonl_file)
    new_draw_no = last_draw_no + 1
    new_lotto_data = fetch_lotto_numbers(new_draw_no)
    
    if new_lotto_data:
        append_to_jsonl(jsonl_file, new_lotto_data)
        print(f"Added draw number {new_draw_no} to {jsonl_file}")
    else:
        print(f"Failed to fetch data for draw number {new_draw_no}")

if __name__ == "__main__":
    main()
