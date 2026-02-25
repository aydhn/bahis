import os
from unittest.mock import patch

# We need to reload the module to pick up env var changes because
# Settings is instantiated at module level in config.py
import src.system.config

def test_neo4j_password_default_none():
    """Verify NEO4J_PASSWORD is None by default (no hardcoded secret)."""
    # Ensure env var is not set
    with patch.dict(os.environ, {}, clear=True):
        # Re-instantiate Settings
        settings = src.system.config.Settings()
        assert settings.NEO4J_PASSWORD is None

def test_neo4j_password_can_be_set():
    """Verify NEO4J_PASSWORD can be set via env var."""
    with patch.dict(os.environ, {"NEO4J_PASSWORD": "secure_password"}, clear=True):
        settings = src.system.config.Settings()
        assert settings.NEO4J_PASSWORD == "secure_password"
