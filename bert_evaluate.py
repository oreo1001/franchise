import json
import argparse
from pathlib import Path
import numpy as np
from evaluate import load

# --- JSON_PATH만 외부 인자로 받음 ---
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="./test_data.json", help="테스트 JSON 파일 경로")
parser.add_argument("--output_json", type=str, default="./bertscore_results.json", help="결과 저장 JSON 파일 경로")
args = parser.parse_args()

json_path = Path(args.json_path).resolve()
output_json = Path(args.output_json).resolve()

def evaluate_bert_score():
    # 1) Gold answers 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [r["question"] for r in data]
    answers = [r["answer"] for r in data]
    ground_truth = [r["ground_truth"] for r in data]

    # 2) BERTScore 계산
    bertscore = load("bertscore")
    results = bertscore.compute(
        predictions=answers,
        references=ground_truth,
        model_type="xlm-roberta-base",
        lang="ko",
        num_layers=12,
        idf=False
    )

    precision = float(np.mean(results["precision"]))
    recall = float(np.mean(results["recall"]))
    f1 = float(np.mean(results["f1"]))

    # 3) 출력
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # 4) 결과 JSON 저장
    result_obj = {
        "json_path": str(json_path),
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    with open(output_json, "a", encoding="utf-8") as f:
        json.dump(result_obj, f, indent=4, ensure_ascii=False)
        
    print(f"{json_path} 평가가 완료되었습니다.")
    
def json_shuffle():
    import json
    import random
    import os

    INPUT_FILE = "/home/sm7540/workspace/franchise/test/2460367501.json"
    OUTPUT_FILE = "/home/sm7540/workspace/franchise/test/2460367501_shuffled.json"

    # 1) 파일 경로 확인
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} 파일을 찾을 수 없습니다.")

    # 2) JSON 로드
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3) 항목이 리스트인지 확인
    if not isinstance(data, list):
        raise ValueError("JSON 최상위 구조가 리스트가 아닙니다.")

    # 4) 항목 섞기
    random.shuffle(data)

    # 5) 결과 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Shuffled JSON이 {OUTPUT_FILE}에 저장되었습니다.")
def print_evaluation_results():
    if not output_json.exists():
        print(f"{output_json} 파일이 존재하지 않습니다.")
        return

    try:
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
            # 결과가 리스트가 아닌 경우에도 처리
            if isinstance(results, dict):
                results = [results]
    except json.JSONDecodeError:
        print(f"{output_json}는 유효한 JSON 형식이 아닙니다.")
        return

    print("\n--- 평가 완료된 결과 목록 ---")
    for idx, result in enumerate(results, 1):
        print(f"[{idx}] 파일: {result.get('json_path', 'N/A')}")
        print(f"   Precision: {result.get('precision', 'N/A'):.4f}")
        print(f"   Recall:    {result.get('recall', 'N/A'):.4f}")
        print(f"   F1:        {result.get('f1', 'N/A'):.4f}")
        print("-" * 40)

if __name__ == "__main__":
    evaluate_bert_score()


