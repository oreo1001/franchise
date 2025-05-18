from typing import List, Any, Callable
import json
from pydantic import BaseModel
import logging
import time
from google.api_core import exceptions as google_exceptions
from google import genai

logger = logging.getLogger(__name__)

class RerankedDocument(BaseModel):
    doc_id: str
    relevance_score: float

class Reranker:
    """Gemini 모델을 사용해 문서를 리랭킹하는 클래스"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """API 키와 모델 이름을 받아 Gemini 클라이언트 초기화"""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("유효한 API 키를 제공해야 합니다.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def build_document_text(self, doc: Any, index: int) -> tuple[str, str, str]:
        """문서의 메타데이터와 내용을 처리하여 헤더, 전체 텍스트, doc_id 반환"""
        metadata = doc.metadata
        doc_id = metadata.get("source", f"missing_id_{index}")  # doc_id 누락 시 대체값
        brand = metadata.get("brand", "")
        company = metadata.get("company", "")
        topic = metadata.get("topic", "")
        sub_topic = metadata.get("sub_topic", "")
        year = metadata.get("year", "")
        metadata_header = f"[{company} | {brand} | {year}년 | {topic} - {sub_topic}]"
        doc_text = doc.page_content.strip()
        full_text = f"{metadata_header}\n{doc_text}"
        return doc_id, full_text 

    def rerank_docs(
        self,
        query: str,
        results: List[Any],
        build_doc_text_callback: Callable = None,
        max_retries: int = 3,
        retry_count: int = 0
    ) -> List[Any]:
        """검색된 문서를 Gemini 모델로 리랭킹"""
        try:
            # 콜백이 없으면 기본 build_document_text 사용
            build_doc_text = build_doc_text_callback or self.build_document_text

            # 1) docs_payload 구성 및 doc_map 생성
            docs_payload = []
            doc_map = {}
            for i, doc in enumerate(results):
                try:
                    _, full_text, doc_id = build_doc_text(doc, i)
                    if doc_id in doc_map:
                        logger.warning(f"중복된 doc_id 발견: {doc_id}")
                    docs_payload.append({"doc_id": doc_id, "content": full_text})
                    doc_map[doc_id] = doc
                except Exception as e:
                    logger.error(f"문서 {i} 처리 중 콜백 에러: {str(e)}")
                    continue

            # 2) 프롬프트 생성 및 API 호출
            prompt = (
                "당신은 문서의 관련성을 평가하는 전문가입니다. "
                "주어진 질문과 문서 목록을 보고, 각 문서가 질문에 얼마나 관련 있는지 0.0에서 1.0 사이의 점수로 평가하세요. "
                "결과는 JSON 형식으로 반환하며, 각 문서의 doc_id와 관련성 점수를 포함해야 합니다. "
                "doc_id는 제공된 값을 정확히 반환해야 하며, 수정하거나 누락시키지 마세요.\n\n"
                f"질문: {query}\n\n"
                "문서 목록:\n"
                f"{json.dumps(docs_payload, ensure_ascii=False, indent=2)}\n"
            )
            response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[RerankedDocument]
            }
            )

            # 3) JSON 파싱 및 doc_id 매핑
            reranked_results = json.loads(response.text)

            # 4) reranked_results의 doc_id 순서대로 정렬된 문서 리스트 반환
            reranked_docs = []
            for result in sorted(reranked_results, key=lambda x: x['relevance_score'], reverse=True):
                doc_id = result['doc_id']
                if doc_id in doc_map:
                    reranked_docs.append(doc_map[doc_id])
                else:
                    logger.warning(f"doc_id {doc_id}가 원본 문서에 없음, 무시됨")

            return reranked_docs

        except google_exceptions.ResourceExhausted as e:
            if "quota" in str(e).lower() or "retry" in str(e).lower():
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count * 30  # 지수 백오프
                    logger.warning(f"쿼터 초과, {wait_time}초 후 재시도합니다... ({retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                    return self.rerank_docs(query, results, build_doc_text_callback, max_retries, retry_count + 1)
                else:
                    logger.error(f"최대 재시도 횟수 {max_retries} 초과")
                    return results
            else:
                logger.error(f"리랭킹 실패 (ResourceExhausted): {str(e)}")
                raise e
        except Exception as e:
            logger.error(f"리랭킹 실패: {str(e)}")
            return results