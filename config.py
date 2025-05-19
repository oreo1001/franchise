# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    VECTOR_DB_PATH: str = "./vector_db/franchise"
    MODEL_NAME: str = "gemini-2.0-flash"
    #MODEL_NAME: str = "gemini-1.5-flash-8b"
    EMBEDDING_MODEL_PATH: str = "nlpai-lab/KURE-v1"
    DEVICE: str = "cuda"
    # DEVICE: str = "cpu"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True  # 대소문자 구분 활성화

# 전역 설정 객체 생성
settings = Settings()