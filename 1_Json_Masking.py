#1_Json_Masking.py
'''
{
  "QUESTION": "㈜틴트어카코리아의 상호명은 무엇인가요?",
  "ANSWER": "상호명은 ㈜틴트어카코리아입니다."
}
>

{
  "QUESTION": "㈜틴트어카코리아의 상호명은 무엇인가요?",
  "ANSWER": "상호명은 ㈜틴트어카코리아입니다.",
  "DEIDENTIFIED_Q": "***의 상호명은 무엇인가요?"
}
'''

import os
import json
import re

# 원본 JSON 파일이 들어 있는 폴더
SOURCE_DIR = "data/train"

# 결과를 저장할 폴더 (없으면 생성)
TARGET_DIR = "data/train_masked"
os.makedirs(TARGET_DIR, exist_ok=True)

# 텍스트에서 민감 정보 패턴을 정규식으로 찾아 *** 로 치환
def mask_sensitive(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, "***", text)
    return text

# 회사명과 브랜드명을 기반으로 정규식 패턴 목록 생성
def build_patterns(company_name, brand_name):
    # (주), ㈜ 제거한 버전도 함께 마스킹 대상에 포함
    company_no_prefix = re.sub(r"^\(주\)|^㈜", "", company_name)

    # 회사명 완전 일치 패턴과 (주) 없는 버전
    company_pattern_1 = re.escape(company_name)
    company_pattern_2 = re.escape(company_no_prefix)

    # 브랜드명 + 괄호 포함 표현 예: "틴트어카(Tint a Car)"
    brand_pattern = re.escape(brand_name) + r"\([^)]*\)"

    # 세 가지 패턴 반환
    return [company_pattern_1, company_pattern_2, brand_pattern]

# QAs 내 각 항목에 대해 DEIDENTIFIED_Q 필드를 추가하는 함수
def add_deidentified_q_to_each_qa(data):
    for item in data:
        try:
            # 회사명, 브랜드명 추출
            company_name = item["JNG_INFO"]["JNGHDQRTRS_CONM_NM"]
            brand_name = item["JNG_INFO"]["BRAND_NM"]

            # 정규식 마스킹 패턴 생성
            patterns = build_patterns(company_name, brand_name)

            # 각 QAs 항목에 대해 비식별화된 QUESTION 추가
            for qa in item["QL"]["QAs"]:
                masked_q = mask_sensitive(qa["QUESTION"], patterns)
                qa["DEIDENTIFIED_Q"] = masked_q

        except KeyError as e:
            print(f"KeyError in item: {e}")
    return data

# 모든 JSON 파일을 순회하며 처리하고 저장
def process_all_files():
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith(".json"):
            source_path = os.path.join(SOURCE_DIR, filename)
            target_path = os.path.join(TARGET_DIR, filename)

            # 원본 JSON 파일 읽기
            with open(source_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"JSON decode error in {filename}")
                    continue

            # 비식별화 처리 적용
            updated_data = add_deidentified_q_to_each_qa(data)

            # 결과 JSON 저장
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            print(f"✅ {filename} 처리 완료 (QAs 항목에 DEIDENTIFIED_Q 추가)")

# 메인 함수 실행
if __name__ == "__main__":
    process_all_files()
