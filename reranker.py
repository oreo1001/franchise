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
    """Gemini 모델을 사용해 문서를 리랭킹하는 class"""
    
    def __init__(self, api_key: str=None, model_name: str = "gemini-2.0-flash"):
        """API 키와 모델 이름을 받아 Gemini 클라이언트 초기화"""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("유효한 API 키를 제공해야 합니다.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def rerank_docs(
        self,
        query: str,
        results: List[Any],
        build_fn: Callable = None,
        retry_count: int = 0,
        k: int = 10
    ) -> List[Any]:
        """검색된 문서를 Gemini 모델로 리랭킹"""
        try:
  
            # 1) docs_payload 구성 및 doc_map 생성
            docs_payload = []
            doc_map = {}
            for i, doc in enumerate(results):
                try:                 
                    doc_id, full_text = build_fn(doc, i)
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
            # logger.info("리랭킹 완료")
            return reranked_docs[:10]

        except google_exceptions.ResourceExhausted as e:
            if "quota" in str(e).lower() or "retry" in str(e).lower():
                max_retries = 3
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count * 30  # 지수 백오프
                    logger.warning(f"쿼터 초과, {wait_time}초 후 재시도합니다... ({retry_count + 1}/{max_retries})")
                    time.sleep(wait_time)
                    return self.rerank_docs(query, results, build_fn, max_retries, retry_count + 1)
                else:
                    logger.error(f"최대 재시도 횟수 {max_retries} 초과")
                    return results
            else:
                logger.error(f"리랭킹 실패 (ResourceExhausted): {str(e)}")
                raise e
        except Exception as e:
            logger.error(f"리랭킹 실패: {str(e)}")
            return results