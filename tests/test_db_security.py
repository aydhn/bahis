
import pytest
from pathlib import Path
from src.memory.db_manager import DBManager
import duckdb

@pytest.fixture
def db_manager(tmp_path):
    """Create a temporary DBManager for testing."""
    db_path = tmp_path / "test.duckdb"
    manager = DBManager(db_path=db_path)
    yield manager
    manager.close()

def test_standard_upsert(db_manager):
    """Test standard upsert with valid columns."""
    data = {
        "match_id": "test_1",
        "home_team": "Team A",
        "away_team": "Team B",
        "home_score": 1
    }
    db_manager.upsert_match(data)

    # Verify insertion
    res = db_manager.get_match("test_1")
    assert len(res) > 0
    assert res["home_team"][0] == "Team A"

def test_unknown_column_upsert(db_manager):
    """Test upsert with unknown column (should be ignored)."""
    data = {
        "match_id": "test_2",
        "home_team": "Team A",
        "away_team": "Team B",
        "unknown_col": 123
    }
    # This should NOT raise an error now
    db_manager.upsert_match(data)

    # Verify insertion
    res = db_manager.get_match("test_2")
    assert len(res) > 0
    # Check that unknown_col was ignored (implicit, since no error)

def test_sql_injection_attempt(db_manager):
    """Test SQL injection attempt via dictionary key."""
    # Attempt to inject SQL into the column name
    payload = "home_team) VALUES ('Team C'); --"
    data = {
        "match_id": "test_3",
        payload: "ignored_value"
    }

    # This should NOT execute the injection and NOT raise syntax error
    # It should simply ignore the invalid key
    db_manager.upsert_match(data)

    # Verify that the match was inserted (if match_id valid) OR handled gracefully
    # Since we only provided match_id and invalid key, match_id is inserted
    res = db_manager.get_match("test_3")
    assert len(res) > 0

    # Verify payload was NOT executed as column
    # We can't easily check internal SQL, but lack of error confirms syntax safety

def test_only_match_id(db_manager):
    """Test upsert with only match_id (edge case)."""
    data = {"match_id": "test_4"}
    db_manager.upsert_match(data)

    res = db_manager.get_match("test_4")
    assert len(res) > 0
