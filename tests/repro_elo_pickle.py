
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.quant.models.elo_glicko_rating import EloGlickoSystem

def test_json_serialization():
    system = EloGlickoSystem()

    # Update some state
    system.update("HomeTeam", "AwayTeam", 2, 1)

    # Save state
    test_file = Path("test_elo_state.json")
    system.save_state(test_file)

    # Verify file is valid JSON
    with open(test_file, "r") as f:
        content = json.load(f)
        assert "elo" in content
        assert "glicko" in content
        assert "processed" in content
        print("JSON structure verified.")

    # Create new system
    new_system = EloGlickoSystem()
    loaded = new_system.load_state(test_file)

    assert loaded is True

    # Verify state
    # HomeTeam won, so rating should be > 1500
    h_team = new_system.elo.get_or_create("HomeTeam")
    assert h_team.rating > 1500

    print("Serialization test passed!")

    # Cleanup
    if test_file.exists():
        test_file.unlink()

if __name__ == "__main__":
    test_json_serialization()
