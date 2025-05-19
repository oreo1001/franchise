from langchain_huggingface import HuggingFaceEmbeddings

# 임베딩 모델 로드
embeddings = HuggingFaceEmbeddings(
    model_name="nlpai-lab/KURE-v1",
    model_kwargs={'device': 'cuda'}
)

# 샘플 텍스트로 임베딩 생성
sample_text = "이것은 테스트 문장입니다."
embedding = embeddings.embed_query(sample_text)

# 임베딩 차원 출력
print(f"임베딩 벡터의 길이: {len(embedding)}")