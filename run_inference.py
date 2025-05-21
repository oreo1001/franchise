from franchise_service import GeminiFranchiseService
from config import settings
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """테스트 데이터에 대한 추론을 실행하고 결과를 저장합니다."""
    try:
        # Gemini API 키 설정
        GEMINI_API_KEY = settings.GEMINI_API_KEY
        
        # 출력 경로 설정
        output_path = "/app/test/test_data.json"
        
        # 서비스 초기화
        logger.info("프랜차이즈 RAG 서비스 초기화 중...")
        rag_service = GeminiFranchiseService(api_key=GEMINI_API_KEY)
        
        # 테스트 데이터 처리
        logger.info("테스트 데이터 추론 시작...")
        results = rag_service.process_test_data()
        
        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"추론 결과가 {output_path}에 저장되었습니다. 총 {len(results)}개 항목.")
        
    except Exception as e:
        logger.error(f"추론 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()