from fastapi import FastAPI
from typing import Union
from pydantic import BaseModel
from backend.chatbot import get_chatbot_reponse
import uuid

app = FastAPI(
    title="Cafe Chatbot API",
    description="Backend service for cafe chatbot"
)

class UserInput(BaseModel):
    user_id: str | None = None ## ya post metounda str ile gelecek ya da uuid4 ile üretilecek
    user_message: str

class AIResponse(BaseModel):
    user_id: str
    ai_message: str

@app.get("/")    # / geldiğinde GET metodu çalışacak
def read_root():
    return {"message":"Welcome to Cafe API "}

@app.post("/chat", response_model=AIResponse) # /chat geliğinde POST metodu çalışacak ve AIResponse dönecek

async def handle_chat(user_input: UserInput):
    user_id = user_input.user_id or str(uuid.uuid4())  
    response_text = await get_chatbot_reponse(user_id, user_input.user_message)   # chatbot.py'daki LLM çağrısını yapan fonksiyon.
    return AIResponse(user_id=user_id, ai_message=response_text)
    

