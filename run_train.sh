set -e

# 1) 첫 번째 인자를 환경 변수로 설정
export WANDB_API_KEY=$1

# 2) 학습 스크립트 실행
accelerate launch --mixed_precision fp16 emb_dpr.py