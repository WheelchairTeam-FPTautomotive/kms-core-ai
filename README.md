# KMS AI Agent — Core AI RAG Engine (Repo 3)

This is the Core AI RAG (Retrieval-Augmented Generation) Engine for the **Traceable Voice Copilot** project. It is managed by **Nguyen Minh Thuan (AI Lead)** and is responsible for ingesting manuals, managing vector indices, chunking token boundaries, and serving low-latency grounded search answers.

---

## Technical Stack

* **Language**: Python 3.12+
* **Package Manager**: `uv`
* **Framework**: FastAPI (exposes internal REST interface on port `8001`)
* **Vector Store & Cloud Services**: AWS OpenSearch Serverless (AOSS), AWS Bedrock (Nova / Titan Embeddings), ChromaDB (fallback)
* **Core Libraries**: `opensearch-py`, `boto3`, `tiktoken` (`cl100k_base`), PyPDF, pydantic-settings, pytest
* **Embedding Model**: AWS Bedrock Titan Embeddings v2 (`amazon.titan-embed-text-v2:0`) / ONNX Local
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
│   └── chroma_db/         # Persistent ChromaDB vector store (local fallback)
├── outputs/               # Solution output json/csv artifacts
├── logs/                  # Rotating log files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── main.py            # API server entrypoint (port 8001)
│   ├── core/
│   │   ├── aws_client.py  # AWS Bedrock & OpenSearch client initializers
│   │   └── config.py      # Centralized pydantic-settings config
│   ├── pipelines/
│   │   ├── bedrock_rag.py # AWS Bedrock + OpenSearch k-NN RAG pipeline
│   │   ├── chunker.py     # 512-token sliding window chunker (64 overlap)
│   │   ├── ingest.py      # Batch PDF vector indexer (ChromaDB / OpenSearch)
│   │   └── solve_problem.py # Router, safety checks & citation mapping
│   └── utils/
│       ├── logger.py      # Console and log rotation
│       └── opensearch_utils.py # OpenSearch index mapping & k-NN query utils
├── tests/
│   ├── test_api.py            # Manual API search endpoint UTF-8 test script
│   ├── test_aws_opensearch.py # AWS OpenSearch index & connection tests
│   ├── test_bedrock_rag.py    # AWS Bedrock embedding & RAG pipeline tests
│   ├── test_chunker.py        # Unit tests for token sliding window chunking
│   └── test_latency.py        # Sub-200ms latency benchmark SLA test
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
# AWS & Vector DB Settings
VECTOR_DB_TYPE=opensearch               # opensearch | chroma
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
OPENSEARCH_ENDPOINT=https://<collection-id>.ap-southeast-2.aoss.amazonaws.com
OPENSEARCH_INDEX=automotive-manuals

# Bedrock Models
BEDROCK_MODEL_ID=global.amazon.nova-2-lite-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# Local ChromaDB Fallback
CHROMA_PATH=data/chroma_db
CHROMA_COLLECTION=automotive_manuals
CHUNK_WINDOW=512
CHUNK_OVERLAP=64
PORT=8001
LOG_LEVEL=INFO
```

---

## Ingestion & Vector Indexing Pipeline

Before running queries, populate the vector collection by running the PDF ingestion pipeline:

```bash
# Ingest PDF manuals into Amazon OpenSearch Serverless (Default)
uv run python src/pipelines/ingest.py --target opensearch

# Ingest PDF manuals into local ChromaDB (Fallback)
uv run python src/pipelines/ingest.py --target chroma
```

To reset and re-index the collection from scratch:

```bash
uv run python src/pipelines/ingest.py --target opensearch --reset
```

> **Pipeline Features:**
> - Automatically parses raw PDF manuals from `data/docs_pdf/`.
> - Resolves document SHA-256 IDs to human-readable manual titles using `data/docs_corpus/mapping.json`.
> - Splits text using `tiktoken` into **512-token chunks** with a **64-token overlap**.
> - Vectorizes chunks using AWS Bedrock Titan Embeddings v2 and indexes into Amazon OpenSearch Serverless.

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

All test suites verify:
- AWS OpenSearch index creation & k-NN retrieval.
- AWS Bedrock Titan embedding generation & Nova/Claude synthesis.
- Sliding window token boundaries (512 tokens with 64 overlap).
- Vector search latency (< 200ms SLA).

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
