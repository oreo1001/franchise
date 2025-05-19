#2_Train_Embeding_ver2.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_chroma import Chroma
import json
import os
from tqdm import tqdm  # tqdm 임포트

# Configuration settings
class Settings:
    EMBEDDING_MODEL_NAME = "./KURE-v1-250518-5-merged"# Finetune model
    DATA_DIR = "data/train_masked"
    COLLECTION_NAME="few_shot_finetune"
    VECTOR_DB_PATH = "./vector_db/few_shot_finetune"

settings = Settings()

# Initialize HuggingFace embeddings (KURE-v1)
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    print("KURE-v1 임베딩 모델 로드 성공")
except Exception as e:
    print(f"KURE-v1 임베딩 모델 로드 실패: {str(e)}")
    exit()

print("임베딩 모델(KURE-v1) 로딩 완료")

# Create vector store directory
vector_db_path = settings.VECTOR_DB_PATH
os.makedirs(vector_db_path, exist_ok=True)

# Function to process JSON files
def process_json_files(directory):
    documents = []
    
    # 디렉토리 존재 여부 확인
    if not os.path.exists(directory):
        print(f"디렉토리를 찾을 수 없습니다: {directory}")
        return documents
    
    # JSON 파일 목록 가져오기
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    
    # JSON 파일 처리 진행바
    for filename in tqdm(json_files, desc="Processing JSON files"):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # 레코드 처리 진행바
            for record in tqdm(data, desc=f"Processing records in {filename}", leave=False):
                ql_data = record.get("QL", {})
                qas = ql_data.get("QAs", [])
                ORIGINAL_TEXT = ql_data.get("ORIGINAL_TEXT", "")  # ORIGINAL_TEXT 추출
                EXTRACTED_SUMMARY_TEXT = ql_data.get("EXTRACTED_SUMMARY_TEXT", "")  # ORIGINAL_TEXT 추출
                ABSTRACTED_SUMMARY_TEXT = ql_data.get("ABSTRACTED_SUMMARY_TEXT", "")  # ORIGINAL_TEXT 추출


                # QA 쌍 처리
                for idx, qa in enumerate(qas):
                    DEIDENTIFIED_Q = qa.get("DEIDENTIFIED_Q", "")
                    QUESTION = qa.get("QUESTION", "")
                    ANSWER = qa.get("ANSWER", "")
                    
                    # Create metadata
                    metadata = {
                        "QUESTION": QUESTION,
                        "ANSWER": ANSWER,
                        "ORIGINAL_TEXT": ORIGINAL_TEXT,  # 메타데이터에 ORIGINAL_TEXT 추가
                        "EXTRACTED_SUMMARY_TEXT": EXTRACTED_SUMMARY_TEXT,
                        "ABSTRACTED_SUMMARY_TEXT": ABSTRACTED_SUMMARY_TEXT

                    }
                    
                    # Create LangChain Document object
                    doc = Document(page_content=DEIDENTIFIED_Q, metadata=metadata)
                    documents.append(doc)
                
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {file_path}")
        except json.JSONDecodeError:
            print(f"JSON 파일 파싱 중 오류 발생: {file_path}")
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    print(f"{len(documents)}개의 문서 객체 생성 완료")
    return documents

# 메인 실행 블록
if __name__ == "__main__":
    documents = process_json_files(settings.DATA_DIR)
    
    # 벡터 스토어 생성 (배치 처리 및 진행바 추가)
    try:
        batch_size = 1000  # 배치 크기 설정
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in tqdm(range(total_batches), desc="Creating vector store"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(documents))
            batch_documents = documents[start_idx:end_idx]
            
            if i == 0:
                # 첫 번째 배치로 새 컬렉션 생성
                vector_store = Chroma.from_documents(
                    documents=batch_documents,
                    embedding=embeddings,
                    collection_name=Settings.COLLECTION_NAME,
                    persist_directory=vector_db_path
                )
            else:
                # 기존 컬렉션에 문서 추가
                vector_store = Chroma(
                    collection_name=Settings.COLLECTION_NAME,
                    embedding_function=embeddings,
                    persist_directory=vector_db_path
                )
                vector_store.add_documents(batch_documents)
        
        print(f"벡터 스토어 생성 완료. 저장 경로: {vector_db_path}")
    except Exception as e:
        print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")