import pytest
from src.pipeline.core import create_default_pipeline

def test_mega_import():
    from unittest.mock import MagicMock
    mock_bot = MagicMock()
    mock_bot.enabled = False
    engine = create_default_pipeline(bot_instance=mock_bot)
    assert engine is not None
