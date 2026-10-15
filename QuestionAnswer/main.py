from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load the QA model
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

class Question(BaseModel):
    context: str
    query: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        result = qa_pipeline({
            "question": question.query,
            "context": question.context
        })
        return {
            "answer": result["answer"],
            "score": result["score"]
        }
    except Exception as e:
        return {"error": str(e)}
