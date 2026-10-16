# MSA AI Search Assistant

A local agentic chatbot. It answers ordinary questions directly and uses DuckDuckGo search only when the model decides current or external information is needed.

This is a **local development application**. It has no authentication and should not be exposed publicly as-is.

## What it does

- Chat through a simple HTML/CSS/JavaScript UI served by FastAPI.
- Use [Ollama](https://ollama.com) with `llama3.2:3b` by default (no paid API keys).
- Let a LangChain agent choose whether to call a `web_search` tool.
- Store conversations and messages in SQLite.
- Return source links only from actual search results.

The model chooses the tool. This is not a keyword router wrapped in agent language.

## Requirements

- Windows 10/11 with PowerShell
- Python 3.11 or newer
- [Ollama](https://ollama.com/download) installed locally
- A tool-calling local model such as `llama3.2:3b`

## Windows PowerShell setup

Run these commands from the `web-search-agent` folder.

### 1. Create a virtual environment

```powershell
cd D:\AI-Agents\web-search-agent
python -m venv .venv
```

If PowerShell blocks script activation for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should show `(.venv)`.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the Ollama model

```powershell
ollama pull llama3.2:3b
```

To use a different local model, set `OLLAMA_MODEL` (see Configuration). The model must support tool calling. If it does not, the API returns a configuration error instead of faking agent behavior.

### 4. Start Ollama if needed

Ollama often runs in the background after install. If `http://127.0.0.1:11434` is not reachable:

```powershell
ollama serve
```

Keep that window open, then use a second PowerShell window for the API server. Confirm the model is present:

```powershell
ollama list
```

### 5. Start FastAPI

From `web-search-agent` with the virtual environment active:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 6. Open the frontend

In your browser, open:

[http://127.0.0.1:8000](http://127.0.0.1:8000)

The UI is served by FastAPI on the same origin. You do not need a separate frontend server.

### 7. Test the chatbot

Try these checks in the UI:

1. `Hello` — should answer without searching.
2. `Explain photosynthesis in simple terms.` — should answer from general knowledge.
3. `What are today's top news headlines?` — should search and show source cards when results are available.
4. Click **New Chat** and confirm a fresh transcript.
5. Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) and confirm Ollama and the model are reported as available.

## Configuration

Copy `.env.example` to `.env` if you want a local file. Environment variables are enough.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2:3b` | Local model tag |
| `DATABASE_PATH` | `data/conversations.db` | SQLite file |
| `MODEL_TIMEOUT_SECONDS` | `60` | Per-model HTTP timeout |
| `SEARCH_TIMEOUT_SECONDS` | `15` | DuckDuckGo search timeout |
| `AGENT_TIMEOUT_SECONDS` | `90` | Overall agent run timeout |
| `CONVERSATION_HISTORY_LIMIT` | `12` | Recent messages sent to the model |
| `MAX_MESSAGE_LENGTH` | `4000` | User message limit |
| `MAX_SEARCH_RESULTS` | `5` | Results per search |
| `MAX_SEARCH_CALLS_PER_RUN` | `3` | Search calls per agent run |
| `AGENT_RECURSION_LIMIT` | `12` | LangGraph execution/iteration cap |
| `MODEL_CALL_LIMIT` | `6` | Model call cap per run |

Example:

```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Tool-calling models

This app requires a model that can call tools. Compatible local examples include `llama3.2:3b`, `llama3.1`, `qwen2.5`, and `mistral-nemo`. If the selected model does not support tools, the API returns `TOOL_CALLING_UNSUPPORTED` and tells you how to pick another `OLLAMA_MODEL`. It will not silently replace the agent with a keyword router.

## Automated tests

Tests mock the model and DuckDuckGo. They do **not** require a live search API or a running Ollama model.

```powershell
cd D:\AI-Agents\web-search-agent
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

Covered behavior:

- Conversation creation and isolation
- Saving and retrieving messages
- Direct answers without search
- Tool invocation and source handling
- Search failure and model failure
- Input validation, unknown conversations, timeouts, and overlapping runs

## Live smoke test (manual)

This is separate from the automated suite and needs Ollama plus network access for DuckDuckGo.

1. Start Ollama and FastAPI as above.
2. Confirm `/health` shows `status: ok`, `model_available: true`, and `supports_tools: true`.
3. Send `Hi` and confirm a normal greeting with no source cards.
4. Send a current-events question such as `What is a major news headline today?`
5. Confirm the answer includes numbered citations and clickable source cards, or clearly reports that search failed.
6. Confirm the browser network tab shows only same-origin `/api/*` calls.

DuckDuckGo can rate-limit unofficial clients. If that happens, wait and retry. The app reports the failure instead of inventing sources.

## Project layout

```text
web-search-agent/
  app/
    main.py
    agent.py
    tools.py
    database.py
    schemas.py
    config.py
    static/
      index.html
      styles.css
      app.js
  tests/
  requirements.txt
  .env.example
  .gitignore
  README.md
```

## Security notes

- There is no login, rate limiting, or public-internet hardening.
- Search snippets are treated as untrusted content, not instructions.
- Only `http` and `https` source links are shown.
- Do not expose this process on `0.0.0.0` or a public host without adding authentication and network controls.
