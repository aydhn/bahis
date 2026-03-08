import pytest
from src.pipeline.core import create_default_pipeline

def test_mega_import():
    engine = create_default_pipeline()
    assert engine is not None
