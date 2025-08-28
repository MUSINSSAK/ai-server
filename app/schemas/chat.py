from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    recommended_ids: List[str]
