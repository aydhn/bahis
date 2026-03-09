import sys
import unittest.mock as mock

sys.modules['loguru'] = mock.Mock()
sys.modules['neo4j'] = mock.Mock()
sys.modules['networkx'] = mock.Mock()

import pytest
from src.memory.neo4j_graph import Neo4jFootballGraph

def test_neo4j_graph_create_player():
    graph = Neo4jFootballGraph()
    graph._init_fallback() # use fallback networkx mode

    # Test create_player directly
    player_data = {
        "name": "Lionel Messi",
        "position": "Forward",
        "team": "Inter Miami",
        "market_value": 35.0
    }

    result = graph.create_player("p10", player_data)
    assert result == True

    # Test _add_player_to_match which calls create_player
    match_player_data = {
        "id": "p7",
        "name": "Cristiano Ronaldo",
        "rating": 8.5,
        "goals": 2
    }

    # Just testing it doesn't crash
    graph._add_player_to_match(match_player_data, "m1", "Al Nassr")
