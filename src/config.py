"""
Configuration module for Fast Track AI Agent
Loads settings from environment variables
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    # Model Deployments
    CHAT_MODEL_DEPLOYMENT: str = os.getenv("CHAT_MODEL_DEPLOYMENT", "chat-heavy")
    EMBEDDING_MODEL_DEPLOYMENT: str = os.getenv("EMBEDDING_MODEL_DEPLOYMENT", "embed-large")

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    KB_DIR: Path = BASE_DIR / "kb"
    VECTOR_STORE_DIR: Path = BASE_DIR / "vector_store"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    STATIC_DIR: Path = BASE_DIR / "static"

    # RAG Configuration
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # LLM Configuration
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.7

    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Ensure directories exist
settings.KB_DIR.mkdir(exist_ok=True)
settings.VECTOR_STORE_DIR.mkdir(exist_ok=True)
settings.UPLOADS_DIR.mkdir(exist_ok=True)
settings.STATIC_DIR.mkdir(exist_ok=True)
