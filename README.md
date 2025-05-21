# 📦 실행 매뉴얼

## 도커 이미지 정보

- 이미지 이름: squad-franchise:latest
- squad-franchise.tar에서 로드

## 실행 명령어 예시

```bash
# 이미지 로드
docker load -i squad-franchise.tar
# 테스트 json 폴더 볼륨 마운트, GEMINI API KEY 넣기
docker run --rm -it --gpus all -v your_test_folder:/app/test -e GEMINI_API_KEY=your_api_key squad-franchise:latest /bin/bash

# 도커 내부에서 bash 파일 실행
bash franchise_RAG.sh /app/test

# 도커 내부에서 train 학습 파일 실행
bash run_train.sh your_wanda_db_API_KEY
```
