import datetime as dt
from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    therapy_plans = relationship("TherapyPlan", back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_type = Column(String(32), default="multimodal", nullable=False)
    status = Column(String(32), default="pending", nullable=False)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="sessions")
    features = relationship("FeatureData", back_populates="session", cascade="all, delete-orphan")
    inference_result = relationship("InferenceResult", back_populates="session", uselist=False, cascade="all, delete-orphan")
    therapy_plans = relationship("TherapyPlan", back_populates="session", cascade="all, delete-orphan")


class FeatureData(Base):
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    modality = Column(String(32), nullable=False)
    features_json = Column(JSONB, nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    session = relationship("UserSession", back_populates="features")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    anomaly_score = Column(Float, nullable=False)
    surprise_score = Column(Float, nullable=True)
    surprise_curve_json = Column(JSONB, nullable=True)
    flags_json = Column(JSONB, nullable=True)
    model_name = Column(String(128), nullable=True)
    model_version = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)

    session = relationship("UserSession", back_populates="inference_result")


class TherapyPlan(Base):
    __tablename__ = "therapy_plans"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    generated_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    plan_json = Column(JSONB, nullable=False)
    retrieved_context_json = Column(JSONB, nullable=True)
    model_name = Column(String(128), nullable=True)
    model_version = Column(String(64), nullable=True)

    user = relationship("User", back_populates="therapy_plans")
    session = relationship("UserSession", back_populates="therapy_plans")


class TherapyPlanFeedback(Base):
    __tablename__ = "therapy_plan_feedback"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("therapy_plans.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer_role = Column(String(32), nullable=False)
    rating = Column(Integer, nullable=True)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, nullable=False)