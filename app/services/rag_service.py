from app.schemas.chat import ChatRequest


def get_dummy_answer(request: ChatRequest) -> str:
    # 실제 RAG 로직 구현할 부분
    print(f"Received query: {request.query}")
    return f"'{request.query}'에 대한 답변입니다. (DUMMY RESPONSE)"
