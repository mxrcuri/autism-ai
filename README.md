# Sparkle Labs: Learn. Grow. Sparkle. ✨

An AI-powered early screening and clinical decision support platform for developmental assessment, featuring **Aster**, an AI adventure guide that gamifies clinical therapy data collection and intervention.

## 🎯 Mission & Scope

**Early screening & clinical decision support only** - Not a diagnostic system. Our platform quantifies task-specific behavioral deviation, aligned with screening practice and early-intervention research, while removing clinical stigma through gamification.

### Scientific Positioning
- **Privacy-First**: Zero raw footage storage with client-side feature extraction
- **Evidence-Based**: Grounded in Early Start Denver Model (ESDM) and DREAM-aligned tasks
- **Scalable**: Lightweight runtime screening with deep temporal models for offline calibration
- **Universal**: A gamified development platform for all children that builds a massive data moat

## 🏗️ Architecture Overview

### Phase 1: Offline Training & Calibration
- **Vision Models**: Task-specific TCN-VAE trained on DREAM dataset
- **Audio Models**: TCN-AE/TCN-VAE trained on CHILDES + ASDBank AAC
- **EEG Models** (optional): TCN-VAE with TUH/DEAP priors

### Phase 2: Runtime Screening System
- **Client-Side Processing**: MediaPipe + Web Audio API for privacy-preserving feature extraction
- **FastAPI Backend**: Deviation estimation with task-relative and norm-relative baselines
- **Temporal Analysis**: Surprise curves and risk indexing

### Phase 3: Clinical Support & Follow-Up
- **Agentic AI Layer**: Multi-agent system for screening, clinical support, therapy planning, and progress monitoring
- **ESDM-Aligned Therapy**: Adaptive difficulty, personalized rewards, gamified sessions

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI with Python 3.11+
- **ML/AI**: PyTorch, scikit-learn, MediaPipe
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: JWT-based auth system
- **API**: RESTful endpoints with automatic OpenAPI documentation

### Frontend
- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI (MUI) with custom theming
- **State Management**: React Query + Context API
- **Real-time**: WebSocket connections for live screening
- **Media Processing**: MediaPipe Web, Web Audio API
- **Routing**: React Router v6

### Infrastructure
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development
- **Monitoring**: Health checks and logging
- **Deployment**: Production-ready with environment configs

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.11+ (for local backend development)

### Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd sparkle-labs
