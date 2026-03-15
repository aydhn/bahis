import sys
from unittest.mock import MagicMock

# Mock loguru before importing the module to avoid ModuleNotFoundError
sys.modules['loguru'] = MagicMock()

import pytest
from src.utils.agent_poll_system import AgentPollSystem, CouncilDecision, AgentVote

@pytest.fixture
def poll_system():
    return AgentPollSystem()

def test_determine_consensus_unanimous_yes(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="EVET"),
            AgentVote(vote="EVET"),
            AgentVote(vote="EVET"),
        ],
        yes_count=3,
        no_count=0,
        undecided_count=0
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "unanimous_yes"
    assert result.council_verdict == "OYNA"
    assert result.consensus_emoji == "🟢🟢🟢"

def test_determine_consensus_unanimous_no(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="HAYIR"),
            AgentVote(vote="HAYIR"),
            AgentVote(vote="HAYIR"),
        ],
        yes_count=0,
        no_count=3,
        undecided_count=0
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "unanimous_no"
    assert result.council_verdict == "PAS GEÇ"
    assert result.consensus_emoji == "🔴🔴🔴"

def test_determine_consensus_majority_yes(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="EVET"),
            AgentVote(vote="EVET"),
            AgentVote(vote="KARASIZ"),
        ],
        yes_count=2,
        no_count=0,
        undecided_count=1
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "majority_yes"
    assert result.council_verdict == "OYNA"
    assert result.consensus_emoji == "🟢🟢🔴"

def test_determine_consensus_majority_no(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="HAYIR"),
            AgentVote(vote="HAYIR"),
            AgentVote(vote="EVET"),
        ],
        yes_count=1,
        no_count=2,
        undecided_count=0
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "majority_no"
    assert result.council_verdict == "PAS GEÇ"
    assert result.consensus_emoji == "🟢🔴🔴"

def test_determine_consensus_split(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="EVET"),
            AgentVote(vote="HAYIR"),
            AgentVote(vote="KARASIZ"),
        ],
        yes_count=1,
        no_count=1,
        undecided_count=1
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "split"
    assert result.council_verdict == "KARARSIZ"
    assert result.consensus_emoji == "🟡🟡🟡"

def test_determine_consensus_all_undecided(poll_system):
    council = CouncilDecision(
        votes=[
            AgentVote(vote="KARASIZ"),
            AgentVote(vote="KARASIZ"),
            AgentVote(vote="KARASIZ"),
        ],
        yes_count=0,
        no_count=0,
        undecided_count=3
    )
    result = poll_system._determine_consensus(council)

    assert result.consensus_type == "split"
    assert result.council_verdict == "KARARSIZ"
    assert result.consensus_emoji == "🟡🟡🟡"
