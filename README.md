# 🧩 Sparkle Labs: Multimodal AI Behavioral Intervention System
### *Aster AI Adventure Guide — Learn. Grow. Sparkle.*

[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

Sparkle Labs is a privacy-first, full-stack application designed for early behavioral intervention (based on the Early Start Denver Model - ESDM). It transforms passive screen time into proactive developmental awareness by gamifying clinical therapy data and utilizing a zero-blocking asynchronous RAG pipeline to generate hyper-personalized daily missions.

---

## 🏗️ Architecture Overview

The platform operates on a strict, unbroken stateful data lineage (**UserSession → FeatureData → InferenceResult → TherapyPlan**) ensuring clinical traceability and deterministic LLM guardrails.

1.  **Multimodal Extraction (Edge)**: Next.js frontend captures user interactions safely via client-side MediaPipe (visual features) and Web Audio APIs, ensuring raw media never leaves the device unnecessarily.
2.  **Deviation Estimation (Backend)**: FastAPI ingests abstract feature vectors, piping them through a custom Temporal Convolutional Network with a Variational Autoencoder (**TCN-VAE**) to calculate behavioral anomalies via Mahalanobis distance.
3.  **Smart Clinical Retrieval (RAG)**: Inference flags trigger a local ChromaDB vector database to retrieve the top clinically validated ESDM exercises using SentenceTransformer embeddings.
4.  **Deterministic Generation**: The native AsyncOpenAI client synthesizes the retrieved context into structured, parent-friendly JSON missions. A strict Python validation layer audits the output and stores both the `plan_json` and `retrieved_context_json` natively in PostgreSQL for hallucination debugging.

---

## 🚀 Key Features

### 🧠 Multimodal Behavioral Analysis
- **Eye Gaze tracking** and **Pose Estimation** via MediaPipe.
- **Speech and vocal pattern analysis** (CHILDES-inspired).
- **Parent questionnaire integration** for holistic assessment.
- **Temporal modeling** via TCN-VAE to capture dynamic behavioral shifts.

### 🔍 Anomaly Detection Engine
- **Mahalanobis distance-based** deviation scoring for rigorous statistical flagging.
- **Temporal window aggregation** to analyze behavior over time.
- **Confidence gating** to reduce false positives by filtering low-quality captures.

### 🤖 RAG-based Therapy Planning
- **Local ChromaDB** vector store pre-loaded with ESDM clinical exercises.
- **Semantic retrieval** using high-dimensional `sentence-transformer` embeddings.
- **Async LLM generation** (GPT-4o-mini) for real-time responsiveness.
- **Structured JSON outputs** for seamless UI rendering.

### 🛡️ Reliability & Safety Layer
- **JSON Schema Validation**: Every AI response is audited via `json.loads` and Pydantic.
- **Hallucination Guardrails**: Prompts are anchored strictly in retrieved clinical text.
- **Deterministic Design**: Optimized prompt engineering ensures consistent, professional advice.

### ⚙️ Core Engineering Excellence
- **Privacy-Preserving Edge Inference**: Real-time tracking executes entirely in the browser. Only lightweight feature vectors reach the backend.
- **Sub-Second AI Orchestration**: Replaces synchronous LLM blocking with an `await`-driven flow, supporting massive concurrency with minimal latency.
- **Closed-Loop Feedback**: Therapy generation is dynamically anchored by historical feedback, ensuring the system learns and adapts.
- **Insight Dashboard**: Professional-grade React visualization of longitudinal behavioral patterns.

---

## 🛠️ Tech Stack

-   **Backend Engine**: Python, FastAPI, SQLAlchemy, asyncpg (PostgreSQL)
-   **GenAI & Vector Storage**: Local ChromaDB, LangChain (Ingestion), AsyncOpenAI (gpt-4o-mini)
-   **Machine Learning**: PyTorch, MediaPipe, OpenCV, TCN-VAE, Librosa
-   **Frontend UI**: Next.js 14, React, TailwindCSS, HSL-tailored premium UI

---

## ⚙️ Detailed Setup & Configuration

### 1. Database Setup (PostgreSQL)
You need a local PostgreSQL instance running.
1. Create a new database:
   ```sql
   CREATE DATABASE behavioural_db;
   ```
2. Ensure you have a user with permission to create tables.

### 2. Environment Configuration (`.env`)
Create a `.env` file in the `backend/` directory:
```env
DATABASE_URL=postgresql+asyncpg://<YOUR_USER>:<YOUR_PASS>@localhost:5432/behavioural_db
OPENAI_API_KEY=sk-proj-your-actual-api-key
SECRET_KEY=generate-a-long-random-string-for-jwt
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### 3. Execution Steps
**Backend:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_chroma_db.py  # Builds the therapy vector index
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

*Developed with a focus on Ethical AI, Clinical Precision, and Data Privacy.*
