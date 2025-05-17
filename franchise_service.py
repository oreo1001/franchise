import logging
import os
import json
import time
import google.generativeai as genai
from google import genai as google_genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
import google.api_core.exceptions as google_exceptions
from pydantic import BaseModel
from typing import Dict, Any,List
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RankedDocument(BaseModel):
        id: str
        content: str
        relevance_score: float
        # metadata: Dict[str, Any]  # 모든 메타데이터를 동적으로 저장

class GeminiFranchiseService:
    """Chroma 기반 RAG와 Gemini를 활용한 추천 서비스"""
    def __init__(self, api_key: str = None):
        # 기본 설정 초기화
        self.initial_system_message = (
            "당신은 프랜차이즈 가맹점주를 위한 전문 Q&A 어시스턴트입니다. "
            "당신은 솔트웨어 주식회사가 제공하는 엔터프라이즈 커스터마이징 AI 챗봇 솔루션 'Stal'입니다. "
            "회사가 제공하는 프랜차이즈 가맹본부 및 가맹점 관련 정보(예: 월매출, 가맹비, 로열티 등)로 임베딩된 테이블 데이터를 사용하여 질문에 답변하세요. "
            "반드시 제공받은 context를 기반으로만 답변해야 하며, 별도로 지식을 추가하거나 재구성해서는 안 됩니다. "
            "만약 질문이 제공된 context와 관련이 없거나, 답변할 정보가 없다면 일반적인 답변을 생성해주세요."
            # "모든 답변은 친절하고 정확하게 작성하되, 추측은 절대 하지 마세요."
        )
        self.vectorstore_search_k = 3
        self.context_max_length = 8000
        
        genai.configure(api_key=api_key)
        self.vector_db_path = settings.VECTOR_DB_PATH
        self.model = genai.GenerativeModel(settings.MODEL_NAME)
        embedding_model_path = settings.EMBEDDING_MODEL_PATH
        
        # 구조화된 데이터처리
        self.model_name = settings.MODEL_NAME
        self.client = google_genai.Client(api_key=api_key)

        # QA 데이터 경로
        self.qa_data_path = os.path.join(os.path.dirname(self.vector_db_path), "qa_data/qa_pairs.json")
        self.qa_data = self._load_qa_data()
        
        # 임베딩 모델 초기화 (로컬 모델 사용)
        logger.info(f"로컬 임베딩 모델 로드 중: {embedding_model_path}")
        self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,  # 로컬 경로 사용
                model_kwargs={
                    'device': settings.DEVICE,
                }
            )
        logger.info("로컬 임베딩 모델 로드 성공")
        
        # Chroma 벡터 스토어 초기화
        self.chroma_vectorstore = self.load_chroma_vectorstore()
    
    def _load_qa_data(self):
        """QA 데이터 로드"""
        try:
            if os.path.exists(self.qa_data_path):
                with open(self.qa_data_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
                logger.info(f"{len(qa_data)}개 QA 쌍 로드 완료")
                return qa_data
            else:
                logger.warning(f"QA 데이터 파일을 찾을 수 없습니다: {self.qa_data_path}")
                return []
        except Exception as e:
            logger.error(f"QA 데이터 로드 중 오류: {str(e)}")
            return []
    
    def load_chroma_vectorstore(self):
        """LangChain Chroma 벡터스토어 로드"""
        try:
            # 절대 경로로 변환
            absolute_path = os.path.abspath(self.vector_db_path)
            
            # 경로 존재 여부 확인
            if not os.path.exists(absolute_path):
                logger.error(f"벡터 스토어 경로가 존재하지 않습니다: {absolute_path}")
                raise FileNotFoundError(f"벡터 스토어 경로가 존재하지 않습니다: {absolute_path}")
            
            logger.info(f"벡터 스토어 로드 시도: {absolute_path}")
            
            # Chroma 벡터스토어 로드
            vectorstore = Chroma(
                persist_directory=absolute_path,
                embedding_function=self.embeddings,
                collection_name="contracts_collection"
            )
            
            # 컬렉션 정보 확인
            collection = vectorstore._collection
            count = collection.count()
            logger.info(f"벡터 스토어 로드 완료: {absolute_path}, 문서 수: {count}")
            
            # 문서가 없는 경우 경고
            if count == 0:
                logger.warning("벡터 스토어에 문서가 없습니다.")
                
            return vectorstore
        except Exception as e:
            logger.error(f"LangChain Chroma 벡터스토어 로드 실패: {str(e)}", exc_info=True)
            raise
    
    ## rerank
    def rerank_docs(self, query: str, results):
        """Chroma 검색 결과를 쿼리와의 관련성에 따라 재순위화하여 Document 리스트로 반환"""
        try:
            docs_payload = [
                {
                    "id": doc.metadata.get("ID", ""),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in results
            ]
            prompt = (
                f"사용자 쿼리: {query}\n\n"
                "아래 문서 목록을 사용자 쿼리와의 관련성에 따라 다시 순위를 매겨주세요.\n"
                "가장 관련성 높은 문서가 먼저 오도록 재정렬하고, 각 문서에 관련성 점수(0~1)를 부여해주세요.\n\n"
                "문서 목록:\n"
                f"{json.dumps(docs_payload, ensure_ascii=False, indent=2)}\n"
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "ranked_documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "JNGHDQRTRS_CONM_NM": {"type": "string"},
                                "BRAND_NM": {"type": "string"},
                                "relevance_score": {"type": "number"},
                                "reasoning": {"type": "string"}
                            },
                            "required": [
                                "id", "content", "JNGHDQRTRS_CONM_NM",
                                "BRAND_NM", "relevance_score", "reasoning"
                            ],
                        }
                    }
                },
                "required": ["ranked_documents"],
            }
        }
            )
            print(response)
            exit(1)
            parsed = response.parsed
            ranked_docs = [RankedDocument(**doc) for doc in parsed.get("ranked_documents", [])]
            
            # RankedDocument를 Document로 변환
            document_list = [
                Document(
                    page_content=doc.content,
                    metadata={**doc.metadata, "relevance_score": doc.relevance_score}
                ) for doc in ranked_docs
            ]
            
            logger.info(f"문서 재순위화 완료: {len(document_list)}개 문서")
            return sorted(document_list, key=lambda d: d.metadata.get("relevance_score", 0.0), reverse=True)

        except Exception as e:
            logger.error(f"문서 재순위화 실패: {str(e)}")
            # 오류 시 원래 results를 그대로 반환 (relevance_score 0.0 추가)
            return [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "relevance_score": 0.0}
                ) for doc in results
            ]

    def retrieve_context(self, query: str) -> str:
        """Chroma로 문서 검색 및 컨텍스트 생성"""
        try:
            # 기본적인 similarity search 사용
            search_results = self.chroma_vectorstore.similarity_search(
                query=query, 
                k=self.vectorstore_search_k
            )
            search_results = self.rerank_docs(query,search_results)
            return
            # 검색된 문서로 컨텍스트 구성
            context = ""
            total_length = 0
            
            for i, doc in enumerate(search_results, 1):
                # 메타데이터 추출
                metadata = doc.metadata
                brand = metadata.get("brand", "")
                company = metadata.get("company", "")
                
                topic = metadata.get("topic", "")
                sub_topic = metadata.get("sub_topic", "")
                year = metadata.get("year", "")
                
                # 메타데이터 헤더 구성
                metadata_header = f"[{company} | {brand} | {year}년 | {topic} - {sub_topic}]"
                
                # 문서 내용 가져오기
                doc_text = doc.page_content.strip()
                
                # 최대 컨텍스트 길이 확인
                full_text = f"{metadata_header}\n{doc_text}"
                doc_length = len(full_text)
                
                if total_length + doc_length > self.context_max_length:
                    logger.info(f"최대 컨텍스트 길이 제한으로 {i-1}개의 문서만 포함")
                    break
                
                context += f"[문서 {i}]\n{full_text}\n\n"
                total_length += doc_length

            logger.info(f"Chroma 검색 완료: {len(search_results)}개 문서, 컨텍스트 길이: {total_length}")
            return context
        except Exception as e:
            logger.error(f"Chroma 검색 실패: {str(e)}")
            return ""

    def answer_question(self, query: str) -> str:
        """사용자 질문에 RAG를 통해 답변"""
        try:
            context = self.retrieve_context(query)
            
            if not context:
                return "검색 결과가 없습니다. 다른 질문을 해주세요."

            prompt = f"""
            {self.initial_system_message}

            컨텍스트:
            {context}

            사용자 질문: {query}

            답변:
            """

            for attempt in range(3):  # 최대 3번 재시도
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except google_exceptions.ResourceExhausted as e:
                    if "quota" in str(e).lower() or "retry" in str(e).lower():
                        wait_time = 60
                        logger.warning(f"쿼터 초과, {wait_time}초 후 재시도합니다...")
                        time.sleep(wait_time)
                    else:
                        raise e  # 그 외의 오류는 바로 처리
            return "요청이 반복적으로 실패했습니다. 잠시 후 다시 시도해주세요."
        
        except Exception as e:
            logger.error(f"질문 답변 실패: {str(e)}")
            return f"죄송합니다, 답변 생성 중 오류가 발생했습니다: {str(e)}"
        
    def process_test_data_for_test(self):
        """테스트 데이터 처리 및 예측 결과 생성"""
        if not self.qa_data:
            logger.error("QA 데이터가 없습니다.")
            return []
        
        # 처음 15개의 QA 항목만 선택
        limited_qa_data = self.qa_data[:15]
        logger.info(f"총 {len(self.qa_data)}개 QA 중 처음 15개만 처리합니다.")
        
        results = []
        
        for i, qa_item in enumerate(limited_qa_data):
            try:
                original_text = qa_item["original_text"]
                question = qa_item["question"]
                
                logger.info(f"질문 {i+1}/15 처리 중: {question}")
                
                # RAG를 통한 답변 생성
                answer = self.answer_question(question)
                
                # 결과 저장
                result = {
                    "original_text": original_text,
                    "question": question,
                    "answer": answer  # LLM 추론 결과
                }
                results.append(result)
                
                # 로깅
                logger.info(f"질문 {i+1}/15 처리 완료")
            
            except Exception as e:
                logger.error(f"질문 {i+1}/15 처리 중 오류: {str(e)}")
                # 오류가 발생해도 계속 진행
                results.append({
                    "original_text": str(qa_item.get("original_text", "")),
                    "question": str(qa_item.get("question", "")),
                    "answer": f"오류 발생: {str(e)}"
                })
        
        logger.info(f"총 {len(results)}개의 질문 처리 완료")
        return results
    
    def process_test_data(self):
        """테스트 데이터 처리 및 예측 결과 생성"""
        if not self.qa_data:
            logger.error("QA 데이터가 없습니다.")
            return []
        
        results = []
        
        for i, qa_item in enumerate(self.qa_data):
            original_text = qa_item["original_text"]
            question = qa_item["question"]
            
            logger.info(f"질문 {i+1}/{len(self.qa_data)} 처리 중: {question}")
            
            # RAG를 통한 답변 생성
            answer = self.answer_question(question)
            
            # 결과 저장
            result = {
                "original_text": original_text,
                "question": question,
                "answer": answer  # LLM 추론 결과
            }
            results.append(result)
            
            # 로깅
            logger.info(f"질문 {i+1} 처리 완료")
        
        logger.info(f"총 {len(results)}개의 질문 처리 완료")
        return results
    