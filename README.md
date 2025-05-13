# 📦 실행 매뉴얼

## 도커 이미지 정보

- 이미지 이름: stal-franchise:latest
- stal-franchise.tar에서 로드

## 실행 명령어 예시

```bash
# 이미지 로드
docker load -i stal-franchise.tar
# 테스트 json 폴더 볼륨 마운트, API KEY 넣기
docker run --rm -it --gpus all -v your_test_folder:/app/test -e GEMINI_API_KEY=your_api_key stal-franchise:latest /bin/bash

# 도커 내부에서 bash 파일 실행
bash franchise_RAG.sh /app/test
```
