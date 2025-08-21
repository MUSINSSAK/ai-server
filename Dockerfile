# 1. 베이스 이미지 선택
# 파이썬 3.11 버전의 가볍고 효율적인 slim 버전을 기반으로 이미지를 만듭니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
# 컨테이너 내에서 명령어가 실행될 기본 경로를 /app으로 설정합니다.
WORKDIR /app

# 3. 의존성 설치
# 먼저 requirements.txt 파일만 복사하여 라이브러리를 설치합니다.
# 이렇게 하면 소스 코드가 변경될 때마다 매번 라이브러리를 새로 설치하는 비효율을 막을 수 있습니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
# 현재 디렉토리(로컬)의 모든 파일을 컨테이너의 /app 디렉토리로 복사합니다.
COPY . .

# 5. 포트 노출
# FastAPI 서버가 8000번 포트에서 실행될 것이므로, 컨테이너의 8000번 포트를 외부와 연결할 수 있도록 개방합니다.
EXPOSE 8000

# 6. 서버 실행 명령어
# 컨테이너가 시작될 때 uvicorn을 사용해 app/main.py 안의 app 객체를 실행합니다.
# --host 0.0.0.0 옵션은 컨테이너 외부에서 접근 가능하도록 만듭니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
