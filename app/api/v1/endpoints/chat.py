from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import initialize_vector_store, get_rag_answer

router = APIRouter()


# 서버 시작 시 벡터 스토어 초기화
@router.on_event("startup")
def startup_event():
    try:
        initialize_vector_store()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize vector store: {e}"
        )


@router.post("/chat", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    # 수정: get_rag_answer 함수 호출
    final_answer = get_rag_answer(request)

    return ChatResponse(answer=final_answer)
