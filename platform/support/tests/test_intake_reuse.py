from unittest.mock import patch
from contextlib import ExitStack

from src.orchestrator import run_sequential_support_workflow


def decision():
    return {
        "ticket": {"summary": "Original intake summary", "category": "VPN Connectivity",
                   "priority": "Medium", "assigned_team": "Network Support"},
        "preliminary_classification": {"category": "VPN Connectivity",
                                       "retrieval_intents": ["vpn"], "retrieval_mode": "targeted"},
    }


def test_injected_intake_skips_all_downstream_classification():
    classification = decision()
    with ExitStack() as stack:
        stack.enter_context(patch("src.orchestrator.classify_ticket", side_effect=AssertionError("Duplicate intake")))
        stack.enter_context(patch("src.generator.classify_ticket", side_effect=AssertionError("Duplicate final classifier")))
        stack.enter_context(patch("src.generator.retrieve_relevant_chunks", return_value=[]))
        result = run_sequential_support_workflow("VPN failure", "it_support", "test", classification)
    assert result["ticket"] == classification["ticket"]
    assert result["preliminary_classification"] == classification["preliminary_classification"]
    assert result["classification_origin"] == "unified_intake"
    assert result["ticket_draft"]["assigned_team"] == "Network Support"


def test_standalone_support_classifies_once_and_reuses_for_retrieval_and_draft():
    with ExitStack() as stack:
        classify = stack.enter_context(patch("src.orchestrator.classify_ticket", return_value=decision()["ticket"]))
        stack.enter_context(patch("src.generator.classify_ticket", side_effect=AssertionError("Duplicate final classifier")))
        stack.enter_context(patch("src.generator.retrieve_relevant_chunks", return_value=[]))
        result = run_sequential_support_workflow("VPN failure", "it_support", "test")
    classify.assert_called_once_with("VPN failure")
    assert result["classification_origin"] == "support"
    assert result["ticket_draft"]["category"] == "VPN Connectivity"
