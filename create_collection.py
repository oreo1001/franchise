from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config import settings
import argparse
import json
import os
import glob

def main():
    # 명령줄 인자 처리 - JSON 경로만 받음
    parser = argparse.ArgumentParser(description='JSON 파일을 임베딩하여 벡터 DB 생성')
    parser.add_argument('--json_path', type=str, required=True, help='JSON 파일 또는 디렉토리 경로')
    args = parser.parse_args()
    
    # JSON 경로 설정
    json_path = args.json_path
    
    # 나머지 설정은 settings에서 가져옴
    vector_db_path = settings.VECTOR_DB_PATH
    device = settings.DEVICE
    model_path = settings.EMBEDDING_MODEL_PATH
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(vector_db_path, exist_ok=True)
    
    # 질문-답변 데이터 저장용 디렉토리
    qa_data_dir = os.path.join(os.path.dirname(vector_db_path), "qa_data")
    os.makedirs(qa_data_dir, exist_ok=True)
    
    # HuggingFace 임베딩 모델 초기화
    print(f"로컬 임베딩 모델 로드 중: {model_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': device}
    )
    print("로컬 임베딩 모델 로드 성공")
    
    # 단일 JSON 파일 또는 디렉토리 처리
    json_files = []
    if os.path.isdir(json_path):
        json_files = glob.glob(os.path.join(json_path, '*.json'))
    else:
        json_files = [json_path]
    
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")
    
    # 문서 객체 저장 리스트 및 QA 데이터
    documents = []
    qa_data = []
    file_count = 0
    document_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                contracts_data = json.load(file)
            
            file_count += 1
            print(f"파일 {os.path.basename(json_file)} 로드 완료 - {len(contracts_data)}개 항목")
            
            for idx, contract in enumerate(contracts_data):
                doc_id = f"{contract['LRN_DTIN_MNNO']}_{idx}"
                
                # 원본 텍스트 저장
                original_text = contract["QL"].get("ORIGINAL_TEXT", "")
                
                # QA 쌍이 있으면 추출하여 저장
                if "QAs" in contract["QL"] and contract["QL"]["QAs"]:
                    for qa in contract["QL"]["QAs"]:
                        qa_item = {
                            "original_text": original_text,
                            "question": qa["QUESTION"],
                            "answer": qa["ANSWER"],
                            "doc_id": doc_id,
                            "brand": contract["JNG_INFO"]["BRAND_NM"],
                            "company": contract["JNG_INFO"]["JNGHDQRTRS_CONM_NM"]
                        }
                        qa_data.append(qa_item)
                
                # 메타데이터 설정 - 원본 텍스트 제외 (컨텐츠에 있으므로 중복 제거)
                metadata = {
                    "ID": contract["LRN_DTIN_MNNO"],
                    "source": doc_id,
                    "brand": contract["JNG_INFO"]["BRAND_NM"],
                    "company": contract["JNG_INFO"]["JNGHDQRTRS_CONM_NM"],
                    "year": contract["JNG_INFO"]["JNG_BIZ_CRTRA_YR"],
                    "topic": contract["ATTRB_INFO"]["KORN_UP_ATRB_NM"],
                    "sub_topic": contract["ATTRB_INFO"]["KORN_ATTRB_NM"],
                    "file_name": os.path.basename(json_file),
                    # "original_text": original_text
                }

                # 콘텐츠 구성 최적화
                content = ""

                # 원본 사용
                if "ORIGINAL_TEXT" in contract["QL"] and contract["QL"]["ORIGINAL_TEXT"]:
                    original_text = contract["QL"]["ORIGINAL_TEXT"]
                    content = f"{original_text}\n\n"
                      # 2. 요약된 내용을 우선적으로 사용 (LLM이 이해하기 쉽게)
                elif "EXTRACTED_SUMMARY_TEXT" in contract["QL"] and contract["QL"]["EXTRACTED_SUMMARY_TEXT"]:
                    extracted_summary = contract["QL"]["EXTRACTED_SUMMARY_TEXT"]
                    content = f"{extracted_summary}\n\n"    

                # 내용이 비어있지 않은 경우만 문서 생성
                if content.strip():
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
                    document_count += 1
                
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {json_file}")
            exit(1)
        except json.JSONDecodeError:
            print(f"JSON 파일 파싱 중 오류가 발생했습니다: {json_file}")
            exit(1)
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {json_file} - {str(e)}")
            exit(1)
    
    print(f"{file_count}개 파일에서 {document_count}개의 문서 객체 생성 완료")
    print(f"{len(qa_data)}개의 질문-답변 쌍 추출 완료")
    
    if not documents:
        print("임베딩할 문서가 없습니다.")
        exit(1)
    
    # QA 데이터 저장
    qa_data_path = os.path.join(qa_data_dir, "qa_pairs.json")
    with open(qa_data_path, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)
    print(f"질문-답변 데이터가 {qa_data_path}에 저장되었습니다.")
    
    # 벡터 스토어 생성 (데이터가 많을 경우 분할 처리)
    batch_size = 100  # 메모리 사용량 고려하여 배치 크기 조정
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    try:
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_documents = documents[start_idx:end_idx]
            
            print(f"배치 {i+1}/{total_batches} 처리 중 ({len(batch_documents)}개 문서)")
            
            # 첫 번째 배치는 새 컬렉션 생성, 이후 배치는 기존 컬렉션에 추가
            if i == 0:
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embeddings,
                    collection_name="contracts_collection",
                    persist_directory=vector_db_path
                )
            else:
                # 기존 벡터 스토어 로드
                vector_store = Chroma(
                    collection_name="contracts_collection",
                    embedding_function=embeddings,
                    persist_directory=vector_db_path
                )
                # 문서 추가
                vector_store.add_documents(batch_documents)
        
        print(f"벡터 스토어 생성 완료. 저장 경로: {vector_db_path}")
        
    except Exception as e:
        print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()