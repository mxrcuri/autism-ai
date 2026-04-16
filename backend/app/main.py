from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from .database import engine, Base

# Routers
from .auth import router as auth_router
from .routes.infer import router as infer_router
from .routes.assessments import router as assessment_router
from .routes.therapy import router as therapy_router  # NEW (you’ll add this)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Autism AI Backend",
    version="1.0.0",
    description="Multimodal behavioral analysis system with async inference and RAG therapy planning"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # update in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(infer_router, prefix="/api/v1", tags=["Inference"])
app.include_router(assessment_router, prefix="/api/v1/assessments", tags=["Assessments"])
app.include_router(therapy_router, prefix="/api/v1/therapy", tags=["Therapy"])  # NEW

@app.on_event("startup")
async def startup():
    logger.info("Starting backend...")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database connected and tables ensured.")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "status": "backend running",
        "version": "1.0.0",
        "routes": [
            "/api/v1/auth/*",
            "/api/v1/infer/video",
            # "/api/v1/infer/audio",
            # "/api/v1/infer/questionnaire",
            "/api/v1/therapy/plan/{session_id}",
            "/api/v1/assessments/*"
        ]
    }