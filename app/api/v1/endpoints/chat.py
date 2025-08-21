from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import rag_service

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    dummy_answer = rag_service.get_dummy_answer(request)
    return ChatResponse(answer=dummy_answer)
