from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

# Fake LLM for testing (no PyTorch, no API key)
class FakeLLM:
    def __call__(self, prompt_text):
        return f"Generated notes for topic: {prompt_text[:50]}..."

llm = FakeLLM()

# Prompt setup (just for consistency)
template = """
Create short, simple, beginner-friendly study notes for the topic:
{topic}
"""
prompt = PromptTemplate(input_variables=["topic"], template=template)

# Chain simulation
class FakeChain:
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm

    def invoke(self, inputs):
        text = self.prompt.format(**inputs)
        return self.llm(text)

chain = FakeChain(prompt, llm)

class Topic(BaseModel):
    topic: str

@app.post("/notes")
async def generate_notes(data: Topic):
    notes = chain.invoke({"topic": data.topic})
    return {"notes": notes}
