from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from langchain_classic.llms import OpenAI
from langchain_classic.chains import RetrievalQA

app = FastAPI()

# Sample product data
products = [
    {"name": "Apple iPhone 14", "details": "Latest iPhone model with A15 chip"},
    {"name": "Samsung Galaxy S23", "details": "Flagship Samsung phone with Snapdragon"},
    {"name": "Dell XPS 13", "details": "High-end ultrabook for business"},
]

# Build vector store
texts = [p["details"] for p in products]
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_texts(texts, embeddings)

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(OpenAI(), retriever=retriever)

class Query(BaseModel):
    question: str

@app.post("/recommend")
async def recommend(query: Query):
    answer = qa_chain.run(query.question)
    return {"recommendation": answer}
