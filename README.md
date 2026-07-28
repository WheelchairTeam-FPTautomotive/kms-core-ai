# KMS AI Agent — Core AI RAG Engine (Repo 3)

This is the Core AI RAG (Retrieval-Augmented Generation) Engine for the **Traceable Voice Copilot** project. It is managed by **Nguyen Minh Thuan (AI Lead)** and is responsible for ingesting manuals, managing vector indices, and serving grounded search answers.

---

## Technical Stack

* **Language**: Python 3.12+
* **Package Manager**: `uv`
* **Framework**: FastAPI (exposes internal REST interface on port `8001`)
* **Core Libraries**: OpenAI, ChromaDB, PyPDF, pydantic-settings, pytest

---

## Prerequisites

Before you begin, ensure the following are installed on your machine:

| Tool | Version | Install |
|------|---------|---------|
| **Python** | 3.12+ | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** | latest | `pip install uv` |
| **Git** | any | [git-scm.com](https://git-scm.com/) |

Verify your setup:

```bash
python --version   # should print Python 3.12.x or higher
uv --version       # should print uv x.x.x
```

---

## Folder Structure

```
kms-core-ai/
├── data/
│   ├── docs_pdf/          # Place PDF manuals here
│   └── chroma_db/         # ChromaDB vector store (auto-generated)
├── outputs/               # Solution output json/csv artifacts
├── logs/                  # Rotating log files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── main.py            # API server entrypoint (port 8001)
│   ├── core/
│   │   └── config.py      # Centralized pydantic-settings config
│   ├── pipelines/
│   │   └── solve_problem.py # Retrieval, safety checks & citation mapping
│   └── utils/
│       └── logger.py      # Console and log rotation
├── scripts/
│   └── run.sh             # Evaluator contract bash wrapper
├── .env.example           # Environment variable template
├── pyproject.toml         # Dependency definitions
└── README.md              # This file
```

---

## Getting Started

### 1. Install Dependencies

```bash
# Run this inside the kms-core-ai/ folder
uv sync
```

This creates a `.venv/` virtual environment and installs all required packages automatically.

### 2. Configure Environment

**Linux / macOS:**

```bash
cp .env.example .env
```

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

Then open `.env` and fill in the required values:

```env
OPENAI_API_KEY=sk-...          # Your OpenAI API key (required for LLM inference)
CHROMA_PATH=data/chroma_db     # Path where ChromaDB will store the vector index
CHROMA_COLLECTION=automotive_manuals  # Name of the ChromaDB collection
PORT=8001                      # Port the FastAPI server listens on
LOG_LEVEL=INFO                 # Logging verbosity: DEBUG | INFO | WARNING | ERROR
```

### 3. Add PDF Manuals

Place all automotive PDF manuals into the `data/docs_pdf/` folder before running the server:

```
kms-core-ai/
└── data/
    └── docs_pdf/
        ├── 2011 - KMS Manual.pdf
        ├── light-control-system.pdf
        └── ...
```

> **Note:** The `data/docs_pdf/` folder is tracked by Git (via `.gitkeep`) but its contents are not. Each team member must obtain and place the PDF files manually.

### 4. Run the Core AI Search Server

```bash
uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001 --reload
```

The server will be available at `http://localhost:8001`.

---

## Core Operations

### RAG Search Query (`POST /api/v1/search`)

Receives query text, checks safety/abstention, executes hybrid vector + keyword lookup, queries LLM, maps page citations, and returns:

```json
{
  "query": "Query text here",
  "answer": "Grounded answer text",
  "citations": [
    {
      "document_id": "...",
      "document_name": "...",
      "section": "...",
      "page": 12,
      "matched_text": "..."
    }
  ],
  "status": "success"
}
```

### Health Check (`GET /api/v1/health`)

```json
{"status": "ready", "service": "kms-core-ai"}
```

### Evaluator Contract (`scripts/run.sh`)

Required script for the automated hackathon scoring system. Evaluates the practice/private test datasets in offline batch mode:

```bash
./scripts/run.sh --input data/public --output outputs/result.json
```
