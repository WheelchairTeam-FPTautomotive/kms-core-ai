# KMS AI Agent — Core AI RAG Engine (Repo 3)

This is the Core AI RAG (Retrieval-Augmented Generation) Engine for the **Traceable Voice Copilot** project. It is managed by **Nguyen Minh Thuan (AI Lead)** and is responsible for ingesting manuals, managing vector indices, chunking token boundaries, and serving low-latency grounded search answers.

---

## Technical Stack

* **Language**: Python 3.12+
* **Package Manager**: `uv`
* **Framework**: FastAPI (exposes internal REST interface on port `8001`)
* **Core Libraries**: ChromaDB, tiktoken (`cl100k_base`), PyPDF, OpenAI, pydantic-settings, pytest
* **Embedding Model**: Default Local ONNX `all-MiniLM-L6-v2` / OpenAI Embedding ($0 cost local inference option)
* **Performance SLA**: **< 150ms** Average Query Latency

---

## Prerequisites

Before you begin, ensure the following are installed on your machine:

| Tool | Version | Install |
| ---- | ------- | ------- |
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

```text
kms-core-ai/
├── data/
│   ├── docs_pdf/          # Raw automotive PDF manuals
│   ├── docs_corpus/       # Transcribed text manuals & mapping.json
│   └── chroma_db/         # Persistent ChromaDB vector store (5,532 chunks)
├── outputs/               # Solution output json/csv artifacts
├── logs/                  # Rotating log files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── main.py            # API server entrypoint (port 8001)
│   ├── core/
│   │   └── config.py      # Centralized pydantic-settings config
│   ├── pipelines/
│   │   ├── chunker.py     # 512-token sliding window chunker (64 overlap)
│   │   ├── ingest.py      # Batch PDF & mapping.json vector indexer
│   │   └── solve_problem.py # Retrieval, safety checks & citation mapping
│   └── utils/
│       └── logger.py      # Console and log rotation
├── tests/
│   ├── test_chunker.py    # Unit tests for token sliding window chunking
│   └── test_latency.py    # Sub-200ms latency benchmark SLA test
├── scripts/
│   └── run.sh             # Evaluator contract bash wrapper
├── .env.example           # Environment variable template
├── pyproject.toml         # Dependency definitions
└── README.md              # Project documentation
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

Then open `.env` and configure the required settings:

```env
OPENAI_API_KEY=sk-...                   # Optional: OpenAI API key
CHROMA_PATH=data/chroma_db              # Persistent ChromaDB storage path
CHROMA_COLLECTION=automotive_manuals   # ChromaDB collection name
CHUNK_WINDOW=512                        # Token window size
CHUNK_OVERLAP=64                        # Token overlap size
USE_LOCAL_EMBEDDING=true                # Use ONNX local embedding model ($0 cost)
PORT=8001                               # FastAPI server port
LOG_LEVEL=INFO                          # Logging verbosity
```

---

## Ingestion & Vector Indexing Pipeline

Before running queries, populate the vector collection by running the PDF ingestion pipeline:

```bash
# Ingest PDF manuals & corpus text into ChromaDB
uv run python src/pipelines/ingest.py
```

To reset and re-index the collection from scratch:

```bash
uv run python src/pipelines/ingest.py --reset
```

> **Pipeline Features:**
> - Automatically parses raw PDF manuals from `data/docs_pdf/`.
> - Resolves document SHA-256 IDs to human-readable manual titles using `data/docs_corpus/mapping.json`.
> - Splits text using `tiktoken` into **512-token chunks** with a **64-token overlap**.
> - Persists embeddings into ChromaDB collection `automotive_manuals`.

---

## Running the Core AI Server

```bash
uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001 --reload
```

The REST API will be available at `http://localhost:8001`.

---

## Testing & Benchmarks

### 1. Run Unit Tests & SLA Latency Benchmarks

```bash
uv run pytest -s
```

All 4 test suites verify:
- Sliding window token boundaries (512 tokens with 64 overlap).
- Vector search latency (< 200ms SLA, average ~148ms).

### 2. Offline Evaluator CLI (`scripts/run.sh`)

Required contract script for the automated hackathon scoring system:

```bash
bash ./scripts/run.sh --input data/test_queries --output output/results.json
```

---

## Core REST Endpoints

### RAG Search Query (`POST /api/v1/search`)

```json
{
  "query": "Hệ thống điều hòa HVAC hoạt động như thế nào?",
  "top_k": 3
}
```

**Response:**

```json
{
  "query": "Hệ thống điều hòa HVAC hoạt động như thế nào?",
  "answer": "Dựa trên tài liệu hướng dẫn kỹ thuật tra cứu được:\n1. [2009 - KMS Manual.pdf - Trang 42]: ...",
  "citations": [
    {
      "document_id": "ce6c4e7562e7fada5e013ead697121e6ebe63f02f6ff5816585f07999b65b8f7",
      "document_name": "2009 - KMS Manual.pdf",
      "section": "Chương 4: HVAC",
      "page": 42,
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
