from sentence_transformers import SentenceTransformer, models

# 1. 기존 transformer 모델 로드
word_embedding_model = models.Transformer('./stal-v2-test', max_seq_length=8192)

# 2. Pooling 설정 (mean-pooling 추천)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# 3. 통합 SentenceTransformer 구성
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 4. 새로운 디렉토리에 저장 (sbert 포맷으로)
model.save('./stal-v2')  # huggingface에 올릴 때도 이걸 기준으로