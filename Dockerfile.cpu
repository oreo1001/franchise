# Stage 1: 빌드
FROM python:3.10-slim AS builder
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: 실행
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local 

COPY config.py franchise_service.py franchise_RAG.sh create_collection.py run_inference.py ./
COPY stal-v1 /app/stal-v1

RUN chmod +x franchise_RAG.sh
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV PYTHONUNBUFFERED=1
CMD ["/bin/bash"]