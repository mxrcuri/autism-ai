# Sparkle Labs: AI-Powered Early Screening 🧩
### *Aster AI Adventure Guide — Learn. Grow. Sparkle.*

[![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

**Sparkle Labs** is a professional-grade clinical decision support system designed to modernize early developmental screening for Autism. By combining **Edge AI (MediaPipe)**, **Temporal Deep Learning (TCN-VAE)**, and **RAG-based Generative AI**, Sparkle Labs transforms subjective clinical observations into objective, privacy-preserving digital biomarkers.

---

## 🌟 Project Vision
Traditional diagnostic methods for developmental delays are often subjective, expensive, and delayed. Sparkle Labs bridges this gap by:
- **Gamifying Assessment**: Removing clinical stigma through "Aster," an AI Adventure Guide.
- **Privacy-First Engineering**: Extracting behavioral features (gaze, pose, motion) entirely in the browser—no raw video ever touches the server.
- **Clinical Alignment**: Grounding therapy recommendations in the **Early Start Denver Model (ESDM)**.

---

## 🏗️ Technical Architecture

### 1. The Inference Pipeline (Vision & Pose)
- **Edge Preprocessing**: Uses MediaPipe Tasks on the client side to extract real-time gaze variance, head pose stability, and motor symmetry.
- **Deep Learning Model**: A **Temporal Convolutional Network (TCN) Autoencoder** on the backend analyzes temporal "surprise."
- **Anomaly Detection**: Quantifies behavioral deviations from normative task baselines (trained on the DREAM dataset).

### 2. RAG-Based Therapy Planning
- **Vector Database**: Clinical exercises from ESDM are vectorized using `sentence-transformers` and stored in **ChromaDB**.
- **Agentic AI**: When an anomaly is detected, a LangChain-style agent retrieves relevant clinical strategies and uses **GPT-4o-mini** to generate a personalized, parent-friendly therapy plan.

### 3. Full-Stack Foundation
- **Backend**: FastAPI with asynchronous database management via **SQLAlchemy** and **AsyncPG**.
- **Auth**: Secure authentication powered by **Argon2** password hashing and **JWT** tokens.
- **Dashboard**: High-performance React (Next.js) dashboard featuring custom-designed assets and real-time telemetry.

---

## 🚀 Getting Started

### Backend Setup
1. **Navigate & Environment**:
   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Database Initialization**:
   - Ensure PostgreSQL is running.
   - Run the ingestion script to build your local therapy knowledge base:
     ```bash
     python scripts/build_chroma_db.py
     ```
3. **Run**:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend Setup
1. **Install & Run**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
2. **Access**: Open `http://localhost:3000/dashboard`

---

## 🛠️ Security & Compliance
- **Zero-Storage Policy**: No raw audio or video is stored.
- **Encryption**: sensitive credentials are managed via environment variables and excluded from Version Control.
- **Audit Trails**: PostgreSQL backend maintains structured clinical records for longitudinal tracking.

---

## 📈 Roadmap & Scientific Positioning
- [ ] Integration of Childes & ASDBank datasets for normative audio prosody.
- [ ] Multi-agent clinical support for therapist collaboration.
- [ ] Longitudinal outcome tracking with surprise curve visualizations.

---

### **Technical Contact**
*Developed by [Your Name]* | [Portfolio] | [LinkedIn]
