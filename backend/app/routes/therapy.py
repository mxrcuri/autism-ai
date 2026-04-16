import json
import os
import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import chromadb
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI

from ..database import get_db
from ..models import UserSession, InferenceResult, TherapyPlan, User
from ..auth import get_current_user

router = APIRouter()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../chroma_data"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

try:
    esdm_collection = chroma_client.get_collection(
        name="esdm_exercises",
        embedding_function=sentence_transformer_ef
    )
except Exception:
    esdm_collection = None

@router.post("/plan/{session_id}")
async def generate_therapy_plan(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if esdm_collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB collection not initialized.")

    result = await db.execute(
        select(UserSession).where(
            UserSession.id == session_id,
            UserSession.user_id == current_user.id
        )
    )
    session = result.scalars().first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or forbidden")

    result = await db.execute(
        select(InferenceResult)
        .where(InferenceResult.session_id == session_id)
        .order_by(InferenceResult.created_at.desc())
    )
    inference = result.scalars().first()

    if not inference:
        raise HTTPException(status_code=400, detail="Inference results not found for this session")

    flags = inference.flags_json or {}
    query_text = f"Anomaly Score: {inference.anomaly_score:.2f}. "
    if flags:
        query_text += "Issues: " + ", ".join([f"{k}: {v}" for k, v in flags.items()])

    try:
        chroma_results = esdm_collection.query(
            query_texts=[query_text],
            n_results=3
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChromaDB retrieval failed: {str(e)}")
    
    retrieved_contexts = chroma_results.get("documents", [[]])[0]
    if not retrieved_contexts:
        raise HTTPException(status_code=500, detail="No context retrieved from ChromaDB")

    context_text = "\n\n".join(retrieved_contexts)

    prompt_instruction = (
        "Act as a pediatric behavioral specialist. "
        "Convert these clinical exercises into a gamified, parent-friendly daily mission schedule addressing the anomaly. "
        "Output ONLY valid JSON."
    )

    try:
        completion = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt_instruction},
                {"role": "user", "content": f"Context:\n{context_text}\n\nAnomalies:\n{query_text}"}
            ],
            response_format={"type": "json_object"}
        )
        
        plan_data = json.loads(completion.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    new_plan = TherapyPlan(
        user_id=current_user.id,
        session_id=session_id,
        plan_json=plan_data,
        retrieved_context_json={"documents": retrieved_contexts},
        model_name="gpt-4o-mini",
        model_version="1"
    )
    
    db.add(new_plan)
    await db.commit()
    await db.refresh(new_plan)

    return {"status": "success", "therapy_plan": plan_data}