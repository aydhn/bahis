from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

class Settings(BaseSettings):
    """Application configuration."""

    # Environment
    ENV: str = "production"
    DEBUG: bool = False
    TELEGRAM_ALLOWED_USERS: str = ""

    # Database
    DB_PATH: str = "data/football.duckdb"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: Optional[str] = None

    # LLM
    LLM_BACKEND: str = "auto"

    # Paths
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    LOG_DIR: Path = ROOT_DIR / "logs"
    DATA_DIR: Path = ROOT_DIR / "data"

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = Settings()
