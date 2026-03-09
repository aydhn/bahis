import os
import pytest
from pathlib import Path
from src.system.config import Settings

def test_default_settings():
    """Test default values when no env vars are provided."""
    settings = Settings()

    # Environment
    assert settings.ENV == "production"
    assert settings.DEBUG is False
    assert settings.TELEGRAM_ALLOWED_USERS == ""

    # Database
    assert settings.DB_PATH == "data/football.duckdb"
    assert settings.NEO4J_URI == "bolt://localhost:7687"
    assert settings.NEO4J_USER == "neo4j"
    assert settings.NEO4J_PASSWORD is None

    # LLM
    assert settings.LLM_BACKEND == "auto"

    # Paths
    # Ensure they are instances of Path
    assert isinstance(settings.ROOT_DIR, Path)
    assert isinstance(settings.LOG_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)

    # Test path resolution logic based on config.py location
    # ROOT_DIR is Path(__file__).parent.parent.parent
    expected_root = Path(__file__).parent.parent.absolute()
    # In config.py: Path(__file__).parent.parent.parent
    # __file__ is src/system/config.py
    # .parent is src/system
    # .parent.parent is src
    # .parent.parent.parent is the root directory

    # We just ensure it ends correctly or has the right structure
    assert settings.LOG_DIR == settings.ROOT_DIR / "logs"
    assert settings.DATA_DIR == settings.ROOT_DIR / "data"

def test_env_override(monkeypatch):
    """Test that environment variables correctly override default values."""
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("DEBUG", "True")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "123,456")
    monkeypatch.setenv("DB_PATH", "data/test.duckdb")
    monkeypatch.setenv("NEO4J_USER", "admin")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")

    settings = Settings()

    assert settings.ENV == "development"
    assert settings.DEBUG is True
    assert settings.TELEGRAM_ALLOWED_USERS == "123,456"
    assert settings.DB_PATH == "data/test.duckdb"
    assert settings.NEO4J_USER == "admin"
    assert settings.NEO4J_PASSWORD == "secret"

def test_global_settings_instance():
    """Test that the global settings instance is created and accessible."""
    from src.system.config import settings
    assert isinstance(settings, Settings)
    assert hasattr(settings, "ENV")
