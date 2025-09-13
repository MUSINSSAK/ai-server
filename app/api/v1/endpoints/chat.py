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


@router.post("/completions", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    # 서비스 함수가 이제 딕셔너리를 반환합니다.
    result = get_rag_answer(request)

    print(f"DEBUG: Returning from AI Server: {result}")

    # 딕셔너리의 각 키를 ChatResponse 모델에 맞게 전달합니다.
    return ChatResponse(
        answer=result["answer"], recommended_ids=result["recommended_ids"]
    )
