# ✅ FILE: main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qa_service import get_answer
import logging

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    sources: list

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query_request: QueryRequest):
    logging.info(f"Received query: {query_request.query}")
    try:
        result, sources = get_answer(query_request.query)
        return QueryResponse(result=result, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {"status": "API is running", "service": "RAG Chatbot"}
