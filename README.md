# 🚀 RAG Agent Assistant: Production-Grade AI Research Engine

A high-performance, resilient, and secure Retrieval-Augmented Generation (RAG) system built with **FastAPI** and **React**. This platform integrates advanced agentic reasoning, hybrid retrieval, cross-encoder reranking, and a multi-layered security architecture to deliver industrial-strength AI capabilities.

---

## 📖 Project Overview

The **RAG Agent Assistant** is designed to solve the problem of information overload in large document sets. Unlike standard chat-with-pdf tools, this system employs an **Agentic Pipeline** that can reason about queries, use specialized tools (like web search or calculators), and validate its own answers against source grounding to prevent hallucinations.

### 🔑 Key Capabilities
*   **Hybrid Retrieval**: Combines BM25 (keyword) and Dense (semantic) search for maximum recall.
*   **Cross-Encoder Reranking**: Utilizes state-of-the-art reranking to ensure only the top-5 most relevant chunks reach the LLM.
*   **Agentic Reasoning**: Powered by LangGraph-style state machines for multi-step problem solving.
*   **Production-Safe Streaming**: Low-latency, ChatGPT-like token streaming with resilient retry logic.
*   **Security-First Design**: Built-in prompt injection protection, input sanitization, and tool sandboxing.

---

## 🏗️ Architecture Overview

The system follows a modern, decoupled micro-services architecture:

*   **Frontend (React)**: A sleek, responsive UI with a real-time streaming chat interface, a live system logs panel, and a document management sidebar.
*   **Backend (FastAPI)**: An asynchronous Python API layer managing orchestration, rate limiting, and session state.
*   **AI Engine**: A modular pipeline leveraging LangChain for orchestration, Groq (Llama 3.1) for lightning-fast inference, and Qdrant Cloud for vector persistence.

### 🗺️ System Architecture Diagram

```plaintext
                                    ┌────────────────────────────────────────────────────────┐
                                    │                AI ORCHESTRATION PIPELINE               │
                                    └────────────────────────────────────────────────────────┘
          ┌─────────────┐           ┌──────────────┐          ┌──────────────────────────────┐
User  ──▶ │  React UI   │  ──HTTP─▶ │ Security     │  ─────▶  │    Query Rewrite (Optional)  │
      ◀── │ (Streaming) │ ◀─SSE───  │ Layer        │          └──────────────┬───────────────┘
          └─────────────┘           └──────────────┘                         ▼
                                           ▲                  ┌──────────────────────────────┐
                                           │                  │    Hybrid Retrieval (D+B25)  │
                                    ┌──────────────┐          └──────────────┬───────────────┘
                                    │ Grounding    │                         ▼
                                    │ Validator    │          ┌──────────────────────────────┐
                                    └──────────────┘          │    Cross-Encoder Reranker    │
                                           ▲                  └──────────────┬───────────────┘
                                           │                                 ▼
          ┌─────────────┐           ┌──────────────┐          ┌──────────────────────────────┐
Final ◀── │  Response   │  ──Agent─▶│ Tools/Memory │  ─────▶  │    Context Compression       │
Answer    └─────────────┘           └──────────────┘          └──────────────────────────────┘
```

---

## 📂 Project Structure

```plaintext
research-assistant/
├── backend/
│   ├── main.py              # Application entry point & Middleware config
│   ├── routes/              # API Endpoints (query, streaming, uploads)
│   ├── core/                # Core AI Logic (Agent, RAG, Reranker, Tools)
│   ├── services/            # Orchestration (Memory, Security, Validation)
│   ├── infra/               # Persistence (Vector DB, Local DB, Embeddings)
│   ├── utils/               # Production Helpers (Caching, Retries, Rate Limits)
│   └── config/              # Environment & Global settings
├── frontend/
│   ├── src/                 # React source (Components, Hooks, Pages)
│   └── public/              # Static assets
├── tests/                   # Smoke tests and unit tests
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── requirements.txt         # Backend dependencies
```

---

## 🔬 The AI Pipeline: Step-by-Step

1.  **Ingress & Validation**: Query is checked for length and sanitized.
2.  **Injection Protection**: Blocklist-based detection stops prompt injection attacks (e.g., \"ignore previous instructions\").
3.  **Dynamic Query Rewriting**: If enabled, the system generates optimized search variants of the user's question.
4.  **Hybrid Retrieval**: Documents are retrieved using both semantic embedding similarity and BM25 linguistic scoring.
5.  **Context Compression**: High-noise sections of retrieved text are compressed or filtered to fit within token limits efficiently.
6.  **Cross-Encoder Reranking**: Documents are re-scored using a deep-learning reranker to sort by actual semantic relevance.
7.  **Agent Reasoning**: The LLM decides whether it can answer directly or needs to call external tools (Calculator, Web Search).
8.  **Memory Injection**: Relevant chat history is retrieved via a sliding window and injected into the prompt context.
9.  **Grounding Validation**: The final answer is cross-referenced against the retrieved documents to ensure factual accuracy.
10. **Safe Streaming**: The response is streamed to the user via Server-Sent Events (SSE) with a fallback to legacy invocation if streaming fails.

---

## 🛠️ Tech Stack

*   **Frontend**: React.js, TailwindCSS (for UI), Framer Motion (for animations).
*   **Backend**: FastAPI, Pydantic v2, Uvicorn.
*   **LLM Inference**: Groq (Llama-3.1-70B/8B).
*   **Vector Database**: Qdrant Cloud.
*   **File Storage**: Supabase Storage.
*   **Caching**: Redis (Persistent).
*   **Orchestration**: LangChain, LangGraph.
*   **Security**: SlowAPI (Rate Limiting), Scikit-Learn (optional for anomaly detection).

---

## 🔒 Security & Reliability

*   **Rate Limiting**: IP-based throttling (10/min for queries, 5/min for uploads) to prevent resource exhaustion.
*   **Data Validation & Sanitization**: Strict input sanitization and context size limiting to prevent overflows and injection.
*   **Tool Sandboxing**: Agent's tool access is restricted to an explicit Allow-List via a dedicated `ToolGuard` service.
*   **Fault Tolerance**: Implemented `SafeStream` wrapper with automatic exponential backoff retries, safe LLM execution wrappers, and pipeline stability measures.
*   **Hallucination Guard**: A grounding validator service scores the LLM's response before it reaches the user.

---

## 🚀 Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   Node.js 18+
*   Groq API Key (Fast Inference)
*   Tavily API Key (Agent Search)

### 2. Backend Setup
```bash
# Navigate to root
cd research-assistant

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Or `.\.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Start the API
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## ⚙️ Configuration & Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
REDIS_URL=redis://localhost:6379
ENABLE_SECURITY=true
ENABLE_CACHE=true
ENABLE_HYBRID=true
```

---

## 🔮 Future Enhancements

*   **Multi-Agent Swarms**: Distributing tasks between a specialist Document Agent and a specialized Web Agent.
*   **LangGraph Studio Integration**: Visualizing agent state transitions in real-time.
*   **Evaluation Framework**: Integrated RAGAS or TruLens for automated retrieval quality assessment.

---

---

## 🚀 Deployment

### 🌐 Backend (Render)
1. Create a new **Web Service** on Render.
2. Connect your GitHub repository.
3. **Runtime**: `Python 3`
4. **Build Command**: `pip install -r requirements.txt`
5. **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
6. **Environment Variables**: Add all variables from the `.env` section.

### 🎨 Frontend (Netlify)
1. Create a new site on Netlify and connect your GitHub repository.
2. **Build command**: `npm run build`
3. **Publish directory**: `dist` (or `build` depending on your setup)
4. **Environment Variables**:
   - `VITE_API_URL`: Your Render backend URL (e.g., `https://your-app.onrender.com`)

---

**Developed with Steve Philip**
