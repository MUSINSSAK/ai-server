from fastapi import FastAPI
from app.api.v1.endpoints import chat

app = FastAPI()

app.include_router(chat.router, prefix="/api/v1")


@app.get("/")
def read_root():
    return {"message": "AI-Server is running"}
