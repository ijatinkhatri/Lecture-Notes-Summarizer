from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "NoteNest — AI Lecture Notes Summarizer"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # LLM Settings
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    DEFAULT_MODEL: str = "gemini-3.5-flash-lite"
    DEFAULT_TEMPERATURE: float = 0.2
    
    # Storage & Uploads
    DATABASE_URL: str = "sqlite+aiosqlite:///./notenest.db"
    SYNC_DATABASE_URL: str = "sqlite:///./notenest.db"
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    
    # Security & CORS
    CORS_ORIGINS: List[str] = ["*"]
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
