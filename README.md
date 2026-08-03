# KMS AI Agent — Core AI RAG Engine (Repo 3)

This is the Core AI RAG (Retrieval-Augmented Generation) Engine for the **Traceable Voice Copilot** project. It is managed by **Nguyen Minh Thuan (AI Lead)** and is responsible for ingesting manuals, managing vector indices, chunking token boundaries, correcting character spacing fragmentation, and serving low-latency grounded search answers.

---

## Technical Stack

* **Language**: Python 3.12+
* **Package Manager**: `uv`
* **Framework**: FastAPI (exposes internal REST interface on port `8001`)
* **Vector Stores**: 
  - **Local**: ChromaDB (with local ONNX `all-MiniLM-L6-v2` embeddings for **$0 Cost & ~10ms Latency**)
  - **Cloud**: AWS OpenSearch Serverless (AOSS) (with Bedrock Titan Embeddings)
* **LLM & Cloud Services**: AWS Bedrock (Nova / Claude), AWS Bedrock Titan Embeddings v2
* **Core Libraries**: `chromadb`, `opensearch-py`, `boto3`, `tiktoken` (`cl100k_base`), `pypdf`, `pydantic-settings`, `pytest`
* **Performance SLA**: **< 150ms** Average Query Latency for local vector searches.

---

## Prerequisites

Before you begin, ensure the following are installed on your machine:

| Tool | Version | Install |
| ---- | ------- | ------- |
| **Python** | 3.12+ | [python.org/downloads](https://www.python.org/downloads/) |
| **uv** | latest | `pip install uv` (or `pipx install uv`) |
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
│   ├── docs_pdf/          # Raw automotive PDF manuals (84 files)
│   ├── docs_corpus/       # Transcribed text manuals & mapping.json
│   └── chroma_db/         # Local persistent ChromaDB vector store
├── output/                # Solution output JSON/CSV artifacts
├── logs/                  # Rotating log files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── main.py            # API server entrypoint (port 8001)
│   ├── core/
│   │   ├── aws_client.py  # AWS Bedrock & OpenSearch client initializers
│   │   └── config.py      # Centralized pydantic-settings config
│   ├── pipelines/
│   │   ├── bedrock_rag.py # AWS Bedrock + OpenSearch/ChromaDB k-NN RAG pipeline
│   │   ├── chunker.py     # 512-token sliding window chunker (64 overlap)
│   │   ├── google_10000_english.txt # English vocabulary wordlist for word merger
│   │   ├── ingest.py      # Batch PDF vector indexer (ChromaDB / OpenSearch)
│   │   └── solve_problem.py # Router, safety checks, & citation mapping
│   └── utils/
│       ├── logger.py      # Console and log rotation
│       └── opensearch_utils.py # OpenSearch index mapping & k-NN query utils
├── tests/
│   ├── test_api.py            # Manual API search endpoint test script
│   ├── test_all_pdfs_api.py   # Spacing issue detection test across all 84 manuals
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

This creates a virtual environment `.venv/` and installs all required dependencies automatically.

### 2. Configure Environment

Create your local `.env` configuration file:

**Linux / macOS:**
```bash
cp .env.example .env
```

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

#### Choose Vector DB Type & Embedding Setup

Edit the `.env` file to select either **Local ChromaDB** or **AWS OpenSearch Serverless**:

##### Option A: Local ChromaDB (Free, $0 Cost, Offline)
```env
VECTOR_DB_TYPE=chroma                  # Set to 'chroma'
USE_LOCAL_EMBEDDING=true               # Uses local default ONNX all-MiniLM-L6-v2 (~10ms latency)
LLM_PROVIDER=none                      # Extractive short driver answer (no LLM)

# Local ChromaDB config
CHROMA_PATH=data/chroma_db
CHROMA_COLLECTION=automotive_manuals

# AWS credentials (still required for Bedrock LLM generation)
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_MODEL_ID=global.amazon.nova-2-lite-v1:0

# Pipeline hyperparameters
CHUNK_WINDOW=512
CHUNK_OVERLAP=64
PORT=8001
LOG_LEVEL=INFO
```

##### Option B: AWS OpenSearch Serverless (Cloud Scale)
```env
VECTOR_DB_TYPE=opensearch              # Set to 'opensearch'
USE_LOCAL_EMBEDDING=false              # Uses AWS Titan Text Embeddings

# AWS Credentials & OpenSearch Settings
AWS_REGION=ap-southeast-2
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
OPENSEARCH_ENDPOINT=https://<collection-id>.ap-southeast-2.aoss.amazonaws.com
OPENSEARCH_INDEX=automotive-manuals

# Bedrock Models
BEDROCK_MODEL_ID=global.amazon.nova-2-lite-v1:0
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0

# Pipeline hyperparameters
CHUNK_WINDOW=512
CHUNK_OVERLAP=64
PORT=8001
LOG_LEVEL=INFO
```

### Unified answer generation (driver-facing)

Retrieval always returns `answer` + `citations`. The **answer** is synthesized for the driver (short, TTS-friendly). Citations stay in the API for evidence UI.

| `LLM_PROVIDER` | Behavior |
|---|---|
| `none` | Extractive 1–2 sentence summary from top chunk (strips `[file.pdf …]` metadata for TTS) |
| `bedrock` | Shared system prompt + params via Bedrock Converse |
| `openai_compatible` | Same prompt/params via OpenAI `/v1/chat/completions` (Ollama, LM Studio, llama.cpp server, vLLM, LocalAI) |

```env
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
OPENAI_API_KEY=ollama
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=400
RAG_TOP_K=3
RAG_CONTEXT_CHARS=2400
RAG_MAX_DISTANCE=1.15
# SYSTEM_PROMPT=...   # optional override
```

### Relevance gate (anti-hallucination)

Chroma returns L2 distances with the default ONNX embedder. Hits with `distance > RAG_MAX_DISTANCE` are discarded. If nothing remains, Core AI returns `status=not_found` and a short TTS-safe refusal — it will **not** invent answers from `pontis.pdf` / `tachonet.pdf` style junk neighbors.

Collection metadata records `hnsw:space=l2`. After a future cosine re-index (`ingest --reset`), retune `RAG_MAX_DISTANCE`.

### Free-talk mode

`POST /api/v1/search` accepts `"mode": "rag" | "free_talk"`. Gateway routes greetings to `free_talk` (no retrieval). With `LLM_PROVIDER=none`, free-talk returns a polite redirect to manual-style questions.

---

## Ingestion & Vector Indexing Pipeline

Before running queries, populate the vector database with the parsed automotive manuals:

```bash
# Ingest PDF manuals into Local ChromaDB (Option A)
uv run python src/pipelines/ingest.py --target chroma

# Ingest PDF manuals into Amazon OpenSearch Serverless (Option B)
uv run python src/pipelines/ingest.py --target opensearch
```

To clear existing collections and re-index everything from scratch:
```bash
uv run python src/pipelines/ingest.py --target chroma --reset
```

> ### 🧠 Smart Text Normalization & PDF Spacing Fixes
> PyPDF's default layout extraction can introduce arbitrary line breaks and character spacing issues (e.g., `progra mme` or `Syste m`). 
> 
> Our ingestion pipeline uses a **Vocab-based Word Joiner** built by Nguyen Minh Thuan (AI Lead) that resolves word spacing fragmentation using a 10,000 English wordlist (`google_10000_english.txt`) + technical keywords. The algorithm processes text in $O(N)$ token cycles to automatically rejoin split terms while leaving legitimate word boundaries (like `in to`) intact.

---

## Running the Core AI Server

Expose the internal REST API on port `8001`:

```bash
uv run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8001 --reload
```

The server REST documentation and health status will be accessible at:
- API Endpoint: `http://localhost:8001/api/v1/search`
- Health check: `http://localhost:8001/api/v1/health`

---

## Testing & Benchmarks

The project comes with a comprehensive suite of unit tests, performance benchmarks, and semantic validation checks.

### 1. Run All Tests
```bash
uv run pytest -s
```
Verify chunking logic, LLM answer synthesis, vector index retrieval, and latency constraints.

### 2. Spacing Quality Gate Verification
Verify that none of the 84 PDF manuals suffer from character/word fragmentation when queried:
```bash
uv run python tests/test_all_pdfs_api.py
```
This script connected to the active vector database, generates technical search queries for each unique PDF manual, and analyzes answers/citations for invalid spaced characters.

### 3. Latency SLA Benchmarking
Benchmark retrieval-augmented generation latency (requires server running):
```bash
uv run python tests/test_latency.py
```

### 4. Offline Evaluator CLI (`scripts/run.sh`)
Execute batch processing for the hackathon evaluator contract:
```bash
bash ./scripts/run.sh --input data/test_queries --output output/results.json
```

---

## Core REST Endpoints

### RAG Search Query (`POST /api/v1/search`)

**Request Payload:**
```json
{
  "query": "Hệ thống điều hòa HVAC được điều khiển qua đâu?",
  "top_k": 3
}
```

**Response:**
```json
{
  "query": "Hệ thống điều hòa HVAC được điều khiển qua đâu?",
  "answer": "Dựa trên tài liệu hướng dẫn kỹ thuật tra cứu được:\n1. [2009 - gaia.pdf - GOG SRS (Trang 11)]: ...\n2. [2005 - microcare.pdf - Trang 7]: ...",
  "citations": [
    {
      "document_id": "ce6c4e7562e7fada5e013ead697121e6ebe63f02f6ff5816585f07999b65b8f7",
      "document_name": "2009 - gaia.pdf",
      "section": "GOG SRS",
      "page": 11,
      "matched_text": "..."
    }
  ],
  "status": "success"
}
```
