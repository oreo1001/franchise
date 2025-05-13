# 📦 실행 매뉴얼

## 도커 이미지 정보

- 이미지 이름: franchise-rag:latest

## 실행 명령어 예시

```bash
# 이미지 로드
docker load -i franchise-rag.tar
# 테스트 json 폴더 볼륨 마운트, API KEY 넣기
docker run --rm \
  -v ${PWD}/test:/app/test \
  -e GEMINI_API_KEY=AIzaSyBJc3CNyKsEEr721vnK1c3Yp7kXud7Pd4U \
  franchise-rag:latest \
  /bin/bash

# 도커 내부에서 bash 파일 실행
bash franchise_RAG.sh /app/test
```
