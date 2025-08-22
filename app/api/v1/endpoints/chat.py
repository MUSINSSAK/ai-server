from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import get_retrieved_documents, initialize_vector_store

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
    # get_retrieved_documents 함수를 호출하여 실제 검색된 문서를 가져옵니다.
    retrieved_docs = get_retrieved_documents(request)

    # 검색된 문서를 하나의 문자열로 합쳐서 응답합니다.
    combined_text = "\n\n---\n\n".join(retrieved_docs)

    # 실제 LLM 응답 대신, 검색된 문서 자체를 반환하여 테스트
    return ChatResponse(answer=f"검색된 관련 문서입니다:\n\n{combined_text}")
