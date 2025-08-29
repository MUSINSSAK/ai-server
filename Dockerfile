# --- Stage 1: Builder ---
# 라이브러리를 설치하고 빌드하는 전용 공간입니다.
FROM python:3.11-slim as builder

# 작업 디렉토리 설정
WORKDIR /app

# 가상환경 생성 (패키지들을 격리하여 관리하기 위함)
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Image ---
# 실제 애플리케이션을 실행할 최종 이미지입니다.
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# Builder 스테이지에서 설치된 가상환경(라이브러리)만 복사해옵니다.
COPY --from=builder /app/venv /app/venv

# 소스 코드를 복사합니다.
COPY . .

# 가상환경의 경로를 PATH에 추가
ENV PATH="/app/venv/bin:$PATH"

# 포트 노출
EXPOSE 8000

# 서버 실행 명령어
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
