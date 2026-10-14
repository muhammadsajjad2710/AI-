import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "business_agent.db"
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "llama3.2:1b"
COMPANY_NAME = "MSA Technologies"
COMPANY_SERVICES = (
    "AI solutions, AI agents, business automation, "
    "Python development, FastAPI, Django, "
    "web applications, and API integration"
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat()


# --------------------------------------------------
# DATABASE
# --------------------------------------------------
def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                contact_name TEXT NOT NULL,
                email TEXT NOT NULL,
                website TEXT,
                industry TEXT,
                need TEXT NOT NULL,
                relevance_score INTEGER,
                analysis_reason TEXT,
                suggested_service TEXT,
                next_action TEXT,
                email_subject TEXT,
                email_body TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lead_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                details TEXT,
                due_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                FOREIGN KEY (lead_id) REFERENCES leads(id)
            )
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def get_lead(lead_id: int) -> dict:
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM leads WHERE id = ?",
            (lead_id,),
        ).fetchone()
    lead = row_to_dict(row)
    if lead is None:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} was not found.")
    return lead


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="MSA Technologies - AI Lead Agent",
    description="Free local AI lead management and automation",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# MODELS
# --------------------------------------------------
class LeadCreate(BaseModel):
    company: str = Field(min_length=1, examples=["ABC Healthcare"])
    contact_name: str = Field(min_length=1, examples=["Ali Khan"])
    email: str = Field(min_length=3, examples=["ali@example.com"])
    website: str = Field(default="", examples=["https://example.com"])
    industry: str = Field(default="", examples=["Healthcare"])
    need: str = Field(
        min_length=1,
        examples=["We need an AI assistant to automate patient appointment inquiries."],
    )


class TaskUpdate(BaseModel):
    status: str = Field(examples=["done"])


# --------------------------------------------------
# LOCAL AI
# --------------------------------------------------
def ollama_is_ready() -> tuple[bool, list[str]]:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return False, []
    models = [item.get("name", "") for item in response.json().get("models", [])]
    return True, models


def model_is_installed(models: list[str]) -> bool:
    return any(name == MODEL or name.startswith(f"{MODEL}:") or name.startswith(MODEL) for name in models)


def parse_model_json(content: str) -> dict:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(
                status_code=502,
                detail="The local model returned invalid JSON. Retry the request.",
            )
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=502,
                detail="The local model returned invalid JSON. Retry the request.",
            )
    if not isinstance(parsed, dict):
        raise HTTPException(
            status_code=502,
            detail="The local model returned invalid JSON. Retry the request.",
        )
    return parsed


def ask_ollama(prompt: str) -> dict:
    ready, models = ollama_is_ready()
    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Open the Ollama app and retry /health.",
        )
    if not model_is_installed(models):
        raise HTTPException(
            status_code=503,
            detail=f"Model {MODEL} is not installed. Run: ollama pull {MODEL}",
        )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You assist {COMPANY_NAME}. "
                    "Reply with one JSON object only. Do not use markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
    except requests.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama connection refused. Open the Ollama app and retry.",
        )
    except requests.Timeout:
        raise HTTPException(
            status_code=504,
            detail="The local model took too long. Close other apps and retry.",
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {exc}")

    content = response.json().get("message", {}).get("content", "")
    return parse_model_json(content)


def as_score(value: object) -> int:
    try:
        score = int(float(str(value)))
    except (TypeError, ValueError):
        score = 50
    return max(0, min(100, score))


def lead_brief(lead: dict) -> str:
    return (
        f"Company: {lead['company']}\n"
        f"Contact: {lead['contact_name']} <{lead['email']}>\n"
        f"Website: {lead['website'] or 'not provided'}\n"
        f"Industry: {lead['industry'] or 'not provided'}\n"
        f"Need: {lead['need']}"
    )


# --------------------------------------------------
# API
# --------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "agent": f"{COMPANY_NAME} AI Lead Agent",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    ready, models = ollama_is_ready()
    return {
        "api": "running",
        "ollama": "up" if ready else "down",
        "model": MODEL,
        "model_installed": model_is_installed(models) if ready else False,
        "installed_models": models,
    }


@app.post("/leads")
def create_lead(payload: LeadCreate):
    now = iso(utc_now())
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO leads (
                company, contact_name, email, website, industry, need,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.company.strip(),
                payload.contact_name.strip(),
                payload.email.strip(),
                payload.website.strip(),
                payload.industry.strip(),
                payload.need.strip(),
                now,
                now,
            ),
        )
        lead_id = cursor.lastrowid
    return {"lead_id": lead_id, "message": "Lead saved."}


@app.get("/leads")
def list_leads():
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM leads ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/leads/{lead_id}")
def read_lead(lead_id: int):
    return get_lead(lead_id)


@app.post("/leads/{lead_id}/analyze")
def analyze_lead(lead_id: int):
    lead = get_lead(lead_id)
    result = ask_ollama(
        "Score how relevant this lead is for "
        f"{COMPANY_NAME}, which offers: {COMPANY_SERVICES}.\n"
        "Use only the information below. Do not invent extra company facts.\n"
        f"{lead_brief(lead)}\n"
        "Return JSON with these keys:\n"
        "relevance_score: integer from 0 to 100\n"
        "reason: one or two sentences\n"
        "suggested_service: one service from the company list\n"
        "next_action: the next human step before any message is sent"
    )
    score = as_score(result.get("relevance_score"))
    reason = str(result.get("reason") or "No reason was returned.").strip()
    service = str(result.get("suggested_service") or "AI solutions").strip()
    next_action = str(
        result.get("next_action") or "Review the lead before outreach."
    ).strip()
    updated = iso(utc_now())
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE leads
            SET relevance_score = ?, analysis_reason = ?, suggested_service = ?,
                next_action = ?, updated_at = ?
            WHERE id = ?
            """,
            (score, reason, service, next_action, updated, lead_id),
        )
    return {
        "lead_id": lead_id,
        "relevance_score": score,
        "reason": reason,
        "suggested_service": service,
        "next_action": next_action,
    }


@app.post("/leads/{lead_id}/draft")
def draft_outreach(lead_id: int):
    lead = get_lead(lead_id)
    result = ask_ollama(
        f"Write a short outreach email from {COMPANY_NAME}.\n"
        f"Services: {COMPANY_SERVICES}.\n"
        "The email is a draft for a human to review. Do not claim it was already sent.\n"
        "Do not invent prices, case studies, or personal details that were not provided.\n"
        f"{lead_brief(lead)}\n"
        "Return JSON with these keys:\n"
        "subject: a concise subject line\n"
        "body: the email body, with a greeting and a closing"
    )
    subject = str(result.get("subject") or f"Idea for {lead['company']}").strip()
    body = str(result.get("body") or "").strip()
    if not body:
        raise HTTPException(
            status_code=502,
            detail="The local model did not return an email body. Retry the request.",
        )

    now = utc_now()
    due = now + timedelta(days=3)
    updated = iso(now)
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE leads
            SET email_subject = ?, email_body = ?, updated_at = ?
            WHERE id = ?
            """,
            (subject, body, updated, lead_id),
        )
        cursor = connection.execute(
            """
            INSERT INTO tasks (lead_id, title, details, due_date, status, created_at)
            VALUES (?, ?, ?, ?, 'pending', ?)
            """,
            (
                lead_id,
                f"Review outreach draft for {lead['company']}",
                "Read the draft and decide whether to send it.",
                due.date().isoformat(),
                updated,
            ),
        )
        task_id = cursor.lastrowid
    return {
        "lead_id": lead_id,
        "subject": subject,
        "body": body,
        "task_id": task_id,
        "due_date": due.date().isoformat(),
    }


@app.get("/tasks")
def list_tasks():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT tasks.*, leads.company, leads.contact_name
            FROM tasks
            JOIN leads ON leads.id = tasks.lead_id
            ORDER BY tasks.due_date ASC, tasks.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.patch("/tasks/{task_id}")
def update_task(task_id: int, payload: TaskUpdate):
    status = payload.status.strip().lower()
    if status not in {"pending", "done"}:
        raise HTTPException(
            status_code=422,
            detail="Status must be 'pending' or 'done'.",
        )
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (status, task_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Task {task_id} was not found.")
        row = connection.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    return dict(row)
