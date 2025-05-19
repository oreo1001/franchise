from langchain_huggingface import HuggingFaceEmbeddings

try:
    embeddings = HuggingFaceEmbeddings(
        # model_name="./stal-v1",
        model_name="./KURE-v1-250518-5-merged",
        model_kwargs={'device': 'cuda'}  # 또는 'cpu'
    )
    print("모델 로드 성공: ./stal-v1")
    
    # 샘플 텍스트로 임베딩 테스트
    sample_text = "이것은 테스트 문장입니다."
    embedding = embeddings.embed_query(sample_text)
    print(f"임베딩 벡터의 길이: {len(embedding)}")
except Exception as e:
    print(f"모델 로드 실패: {str(e)}")