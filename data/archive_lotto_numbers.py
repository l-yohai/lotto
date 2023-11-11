import requests
import json

from tqdm import tqdm

def fetch_lotto_numbers(draw_no):
    # 대한민국 로또 API URL
    url = f'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}'
    
    # API 요청
    response = requests.get(url)
    
    if response.status_code == 200:
        # JSON 데이터 파싱
        return response.json()
    else:
        return None

def save_lotto_numbers_to_jsonl(start_draw, current_draw):
    # 로또 번호들을 저장할 리스트
    lotto_numbers_list = []

    # 1회차부터 현재까지의 로또 번호 가져오기
    for draw_no in tqdm(range(start_draw, current_draw + 1)):
        lotto_data = fetch_lotto_numbers(draw_no)
        if lotto_data:
            lotto_numbers_list.append(lotto_data)
    
    # JSONL 파일로 저장
    with open('data/lotto_numbers.jsonl', 'w') as file:
        for lotto_data in lotto_numbers_list:
            json_record = json.dumps(lotto_data)
            file.write(json_record + '\n')

    return "Lotto numbers saved to lotto_numbers.jsonl"

# 시작 회차와 현재 회차 설정 (예: 1부터 1092회차까지)
start_draw = 1
current_draw = 1092

# JSONL 파일로 저장
save_result = save_lotto_numbers_to_jsonl(start_draw, current_draw)
save_result

