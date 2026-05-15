## AI Research Assistant: The Ultimate Agentic RAG Platform

![AI Research RAG Assistant](docs/images/hero.png)

A high-performance, resilient, and secure **Retrieval-Augmented Generation (RAG)** engine built with **FastAPI** and **React**. This platform integrates advanced **LangGraph** orchestration, hybrid retrieval, and a multi-layered verification system to deliver industrially reliable AI research.

---

## 📖 Project Overview

The **AI Research Assistant** solves the "Information Overload" problem by transforming static document sets into a conversational, authoritative knowledge base. Unlike standard chat-with-pdf tools, this system uses a **State-Driven Agentic Pipeline** that can reason about queries, fetch data via hybrid search, and validate every claim against source grounding.

---

## 🚀 Key Features: Enterprise-Grade RAG

### 1. Multi-User Isolation & Security
Built for multi-tenant environments, the platform ensures total data privacy:
*   **Session-Scoped Data**: Every vector point and storage asset is tagged with a unique `user_id` / `session_id`.
*   **Strict Filtering**: Retrieval and tool calls are forced to filter by `user_id`, preventing cross-user data leakage.
*   **Isolated Storage**: Files are stored in session-specific cloud paths (`uploads/{session_id}/`) on Supabase.

### 2. Resource Protection & Guardrails
To ensure stability and fair usage, the system implements:
*   **Document Caps**: Maximum of **5 documents per session** to prevent database bloat.
*   **Size Constraints**: Strict **10MB per file** limit for uploads.
*   **Rate Limiting**: Integrated `SlowAPI` to prevent abuse (3 uploads/min, 5 queries/min).

### 3. Automated Data Lifecycle & Maintenance
*   **Background Maintenance**: A dedicated `maintenance_service.py` runs periodic consistency audits between the document registry and the vector store.
*   **Vector Audit**: Detects and remediates false-positive "missing vectors" flags by querying Qdrant with specific `doc_id` tags.
*   **TTL Purge**: Automated background jobs delete session vectors and artifacts older than **2 hours**.

### 4. Premium UX with Real-Time Feedback
*   **Modern Sidebar**: Refactored sidebar with better navigation and session management.
*   **Toast Notifications**: High-visibility notification system for errors (429, 413, network), successes, and server health status.
*   **Live Stream Dashboard**: Real-time telemetry of the agent's internal reasoning steps and retrieval logs.

---

## 🏗️ Technical Architecture: The Agentic Core

### 1. LangGraph State Machine
The core reasoning engine is built using **LangGraph**, providing a deterministic and reliable alternative to chaotic LLM loops.
*   **Agent Node**: The LLM's brain—decides between answering directly or calling tools.
*   **Tools Node**: The LLM's hands—executes vector searches and web queries.
*   **Cyclic Control**: The system continuously loops between nodes until a high-confidence answer is synthesized.

```mermaid
graph TD
    User([User Query]) --> Security[Security Guard & Injection Check]
    Security --> Cache{Response Cache?}
    Cache -->|Hit| FinalAnswer
    Cache -->|Miss| Agent[LangGraph Agent Node]
    Agent --> ToolCall{Decision}
    ToolCall -->|Retrieve Facts| Tools[Tools Node]
    Tools --> Telemetery[Real-Time Telemetry]
    Telemetery --> Agent
    ToolCall -->|Synthesize| Validator[Grounding Validator]
    Validator --> FinalAnswer([Final Answer])
```

### 2. Infrastructure & Persistence
*   **Vector DB (Qdrant)**: High-speed semantic search with session-aware payload filtering.
*   **Cloud Storage (Supabase)**: Secure file storage using signed URLs for authenticated access.
*   **Persistence (PostgreSQL)**: Robust document registry using Supabase Postgres to track ingestion states.
*   **Caching (Upstash Redis)**: Distributed caching for session metadata and rate limiting.

### 3. Retrieval Intelligence (Hybrid Search)
*   **Dense Search (Qdrant)**: Captures semantic meaning via high-dimensional vector embeddings.
*   **Keyword Search (BM25)**: Captures exact technical terms, IDs, and proper nouns.
*   **Context Compression**: LLM-powered summarization of retrieved chunks ensures only high-density facts are passed to the context window.

---

## 🛡️ Reliability & Safety Guardrails

### 1. Multi-Tier Grounding Validator
To eliminate hallucinations, every answer passes through a rigorous **Grounding Validator**:
*   **Keyword Overlap**: Initial check for word-level consistency.
*   **Substring Match**: Immediate pass if the answer is a direct quotation.
*   **Semantic Check**: A dedicated LLM verifies if the generated claim is supported by the source material.

### 2. Execution Resilience & Telemetry
*   **SafeStream**: A custom wrapper for **Server-Sent Events (SSE)** that delivers tokens reliably via chunk-aware delivery.
*   **Real-Time Telemetry**: Internal state transitions (Retrieved, Validated, Compressed) are emitted via structured `emit_log`.
*   **Timeout & Retries**: All LLM and tool calls implement a **10s timeout** and automated exponential backoff.

---

## 📂 Project Structure

```plaintext
research-assistant/
├── backend/              # FastAPI Orchestrator
│   ├── Dockerfile        # Production build context
│   ├── core/             # Agentic Brain (LangGraph, Reranker, Telemetry)
│   ├── services/         # Intelligence (Validation, Self-Healing, Maintenance)
│   ├── infra/            # Persistence (Qdrant, Supabase, Redis)
│   ├── routes/           # API Endpoints
│   └── scripts/          # Migration & Reset tools
├── frontend/             # Vite/React UI
│   ├── Dockerfile        # Production build context
│   └── src/              # React source (Hooks, Components, Pages)
├── docker-compose.yml    # Full-stack orchestration
├── render.yaml           # Deployment manifest
└── supabase_migration.sql # Database schema
```

---

## 🛠️ Tech Stack

*   **Backend**: FastAPI, LangGraph, Pydantic v2
*   **Frontend**: React, Vite, Tailwind CSS
*   **AI Models**: Groq (Llama 3.1 8B), Sentence-Transformers (Local Embeddings)
*   **Databases**: Qdrant (Vector), Supabase Postgres (Relational)
*   **Caching**: Upstash Redis
*   **DevOps**: Docker, Docker Compose

---

## 🚀 Setup & Installation

### Option A: Docker (Recommended)
The fastest way to get started is using Docker Compose:
```bash
docker-compose up --build
```
The backend will be available at `http://localhost:8000` and the frontend at `http://localhost:3000`.

### Option B: Local Development

#### 1. Backend Setup
```bash
cd backend
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory (or use individual `.env` files in `backend/` and `frontend/`):

*   `GROQ_API_KEY`: Groq Cloud API Key
*   `TAVILY_API_KEY`: Tavily Search API Key
*   `QDRANT_URL` & `QDRANT_API_KEY`: Qdrant Cloud credentials
*   `SUPABASE_URL` & `SUPABASE_KEY`: Supabase project details
*   `UPSTASH_REDIS_REST_URL` & `UPSTASH_REDIS_REST_TOKEN`: Redis caching credentials

---

## 🆕 May 2026: Latest Upgrades

### Infrastructure & DevOps
- **Docker Support**: Full containerization for both backend and frontend with multi-stage builds.
- **Supabase Integration**: Switched to signed URLs for secure storage access and Postgres-backed document registry.
- **Redis Caching**: Integrated Upstash Redis for distributed session management.

### Resilience & Maintenance
- **Consistency Audit Service**: Added `full_consistency_audit` to resolve orphaned documents and vector synchronization issues.
- **Deduplication Engine**: MD5-based file hashing combined with `doc_id` vector tagging ensures zero-duplicate ingestion.
- **Self-Healing v2**: Improved evaluation scoring and adaptive retrieval strategies.

### UI/UX Improvements
- **Sidebar Modernization**: Refined navigation and session controls.
- **Production Hardening**: Comprehensive security updates including XSS protection and strict input validation.

---

**Developed with 💡 for High-Accuracy Environments by Steve Philip**
