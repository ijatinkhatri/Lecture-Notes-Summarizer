from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from backend.app.core.config import settings
from backend.app.db.database import get_db
from backend.app.db.models import AppSettings

router = APIRouter(prefix="/settings", tags=["Application Settings"])

class SettingsUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    default_model: Optional[str] = None
    temperature: Optional[float] = None
    default_summary_mode: Optional[str] = None

@router.get("")
async def get_settings(db: AsyncSession = Depends(get_db)):
    """Returns application configuration state, model options, and masked API key."""
    api_key = settings.GEMINI_API_KEY
    masked_key = ""
    has_key = bool(api_key and len(api_key) > 5)
    if has_key:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"

    return {
        "has_gemini_api_key": has_key,
        "gemini_api_key_masked": masked_key,
        "active_model": settings.DEFAULT_MODEL,
        "temperature": settings.DEFAULT_TEMPERATURE,
        "default_summary_mode": "short",
        "default_audience": "intermediate",
        "offline_available": True,
        "available_models": [
            {"id": "gemini-3.5-flash-lite", "name": "Gemini 3.5 Flash Lite (Fast & Active Quota, Recommended)"},
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Latest Flagship)"},
            {"id": "gemini-3.1-flash-lite", "name": "Gemini 3.1 Flash Lite"},
            {"id": "offline-extractive-v1", "name": "Offline Extractive Engine (TextRank + YAKE - No API key needed)"}
        ]
    }

@router.post("")
async def update_settings(
    payload: SettingsUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Updates runtime configuration settings."""
    if payload.gemini_api_key is not None:
        clean_key = payload.gemini_api_key.strip()
        settings.GEMINI_API_KEY = clean_key
        # Update or insert in DB
        setting_rec = await db.get(AppSettings, "GEMINI_API_KEY")
        if setting_rec:
            setting_rec.value = clean_key
        else:
            db.add(AppSettings(key="GEMINI_API_KEY", value=clean_key))

    if payload.default_model is not None:
        settings.DEFAULT_MODEL = payload.default_model

    if payload.temperature is not None:
        settings.DEFAULT_TEMPERATURE = payload.temperature

    await db.commit()

    has_key = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 5)
    masked = f"{settings.GEMINI_API_KEY[:4]}...{settings.GEMINI_API_KEY[-4:]}" if has_key else ""

    return {
        "status": "success",
        "has_gemini_api_key": has_key,
        "gemini_api_key_masked": masked,
        "active_model": settings.DEFAULT_MODEL,
        "temperature": settings.DEFAULT_TEMPERATURE
    }
