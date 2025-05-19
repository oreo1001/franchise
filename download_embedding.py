from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
import os

def download_and_convert_model(model_name, save_path):
    """HuggingFace 모델을 다운로드하고 SentenceTransformer로 변환하여 저장"""
    try:
        print(f"모델 {model_name}을 다운로드하여 {save_path}에 저장 중...")
        
        # XLM-RoBERTa 모델 로드
        word_embedding_model = models.Transformer(model_name)
        
        # 풀링 레이어 추가 (mean pooling)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        
        # SentenceTransformer 모델 생성
        sentence_transformer = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        # 로컬 경로에 저장
        os.makedirs(save_path, exist_ok=True)
        sentence_transformer.save(save_path)
        print(f"SentenceTransformer 모델이 {save_path}에 성공적으로 저장되었습니다.")
        
        # 토크나이저도 저장 (HuggingFaceEmbeddings 호환성 보장)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        
    except Exception as e:
        print(f"모델 다운로드 및 변환 실패: {str(e)}")
        exit(1)

if __name__ == "__main__":
    model_name = "nlpai-lab/KURE-v1"
    save_path = "./stal-v1"
    download_and_convert_model(model_name, save_path)