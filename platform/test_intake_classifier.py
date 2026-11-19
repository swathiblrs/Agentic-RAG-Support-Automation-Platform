from unittest.mock import Mock

import api
from intake_classifier import classify_intake


def details():
    return {
        "ticket": {"category": "VPN Connectivity", "priority": "Medium", "assigned_team": "Network Support",
                   "summary": "VPN failure", "ml_model": "test", "ml_category_confidence": 0.9},
        "preliminary_classification": {"category": "VPN Connectivity", "retrieval_intents": ["vpn"],
                                       "retrieval_mode": "targeted", "confidence": 0.9},
    }


def test_it_intake_has_domain_and_ticket_details_from_one_call():
    classifier = Mock(return_value=details())
    result = classify_intake(api.Request(question="VPN stopped working"), classifier)
    classifier.assert_called_once_with("VPN stopped working")
    assert result["domain"] == "it"
    assert result["subcategory"] == "VPN Connectivity"
    assert result["priority"] == "Medium"
    assert result["suggested_team"] == "Network Support"
    assert result["support_classification"]["preliminary_classification"]["retrieval_intents"] == ["vpn"]


def test_pure_production_has_no_forced_it_category():
    classifier = Mock(side_effect=AssertionError("Do not classify a production incident as IT"))
    result = classify_intake(api.Request(question="Checkout returns HTTP 503"), classifier)
    assert result["domain"] == "production"
    assert result["subcategory"] is None
    classifier.assert_not_called()


def test_clarification_does_not_call_ticket_model():
    classifier = Mock()
    result = classify_intake(api.Request(question="Something is broken"), classifier)
    assert result["workflow_path"] == "clarify"
    classifier.assert_not_called()


def test_support_consumes_the_exact_intake_decision(monkeypatch):
    classifier = Mock(return_value=details())
    call = Mock(return_value={
        "answer": "VPN guidance", "ticket_draft": {"category": "VPN Connectivity",
        "priority": "Medium", "assigned_team": "Network Support"},
    })
    monkeypatch.setattr(api.worker, "classify", classifier)
    monkeypatch.setattr(api.worker, "call", call)
    result = api.execute(api.Request(question="VPN stopped working"))
    classifier.assert_called_once()
    call.assert_called_once_with("VPN stopped working", result["classification"]["support_classification"])
    assert result["classification"]["subcategory"] == result["ticket_draft"]["category"]
    assert not result["support"]["delivery"]["public_faq_cache_hit"]


def test_real_support_reuses_intake_without_classifying_again():
    question = "My VPN connection failed"
    classification = api.worker.classify(question)
    assert classification["ticket"]["category"] == "VPN Connectivity"
    # Sentinel category would be replaced if the worker re-ran ticket classification.
    classification["ticket"]["summary"] = "INTAKE-REUSED"
    result = api.worker.call(question, classification)
    assert result["classification_origin"] == "unified_intake"
    assert result["ticket"]["summary"] == "INTAKE-REUSED"
    assert result["preliminary_classification"] == classification["preliminary_classification"]
