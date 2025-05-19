set -e

# 학습 스크립트 실행
accelerate launch --mixed_precision fp16 emb_dpr.py "$@"