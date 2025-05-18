docker build -t stal-franchise-v1:3.0.0 -f Dockerfile .
docker tag stal-franchise-v1:4.0.0 stal-franchise-v1:latest
docker load -i stal-franchise.tar # 테스트 json 폴더 볼륨 마운트, API KEY 넣기 docker run --rm -it --gpus all -v your_test_folder:/app/test -e GEMINI_API_KEY=your_api_key stal-franchise:latest /bin/bash # 도커 내부에서 bash 파일 실행 bash franchise_RAG.sh /app/test
docker save -o stal-franchise-v1-3.0.0.tar stal-franchise-v1:latest
docker run --rm -it --gpus all -v /home/sm7540/project/franchise/test:/app/test -e GEMINI_API_KEY=AIzaSyAgPaXsKKIzBJL4XrHDEm4OWdrj0o4M9G4 stal-franchise-v1:latest /bin/bash # 도커 내부에서 bash 파일 실행 bash franchise_RAG.sh /app/test
bash franchise_RAG.sh /app/test