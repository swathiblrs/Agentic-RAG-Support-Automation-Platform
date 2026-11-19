import json
from types import SimpleNamespace

import httpx
import pytest

import api
from app.models.generic_state import GenericIncidentState
from app.models.schemas import IncidentInput
from app.services.local_analysis import analyze, parse_evidence, redact
from app.services.generic_incident_graph import GenericIncidentGraph


def state():
    return parse_evidence(GenericIncidentState(incident=IncidentInput(
        title="Submission service failed", severity="high",
        logs=['{"level":"ERROR","request_id":"r1","message":"database timeout","status":504}',
              'ERROR HTTP 503 request_id=r1 token=secret'],
        metrics={"cpu_percent": 12},
    )))


def answer(**changes):
    return {
        "summary": "The submitted logs report a timeout; the cause is not confirmed.",
        "severity": "low", "severity_evidence_ids": ["log-1"],
        "observations": [{"text": "A timeout was reported.", "evidence_ids": ["log-1"]}],
        "possible_causes": [{"text": "A dependency may be slow.", "evidence_ids": ["log-1"]}],
        "next_steps": ["Review dependency health for this request."],
        "missing_evidence": ["Dependency health measurements"], **changes,
    }


def transport(output=None, done=True):
    def handle(request):
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})
        body = json.loads(request.content)
        assert body["think"] is False
        assert "local_simulation" not in body["prompt"]
        return httpx.Response(200, json={"done": done, "response": json.dumps(output or answer())})
    return httpx.MockTransport(handle)


def test_parser_extracts_fields_and_correlates_without_inventing_thresholds():
    parsed = state().enrichment["parsed_evidence"]
    assert parsed["request_groups"]["r1"] == ["log-1", "log-2"]
    assert parsed["records"][1]["status"] == "504"
    assert parsed["records"][-1]["interpretation"] == "No service threshold supplied."
    assert "secret" not in parsed["records"][2]["message"]


def test_redaction():
    assert "hunter2" not in redact("password=hunter2 user@example.com Bearer abcdef")
    assert "example.com" not in redact("user@example.com")


def test_local_analysis_checks_citations_and_preserves_severity(monkeypatch):
    monkeypatch.setenv("LOCAL_QWEN_MODEL", "qwen3:1.7b")
    result = analyze(state(), transport()).enrichment["diagnostic_analysis"]
    assert result["status"] == "completed"
    assert result["effective_severity"] == "high"
    assert result["human_review_required"]


@pytest.mark.parametrize("output", [
    answer(severity_evidence_ids=["invented"]),
    answer(observations=[{"text": "Confirmed diagnosis", "evidence_ids": []}]),
    answer(severity="catastrophic"),
])
def test_invalid_model_outputs_are_not_accepted(output):
    result = analyze(state(), transport(output)).enrichment["diagnostic_analysis"]
    assert result["status"] == "unavailable"
    assert "assessment" not in result


def test_truncated_output_is_not_accepted():
    assert analyze(state(), transport(done=False)).enrichment["diagnostic_analysis"]["status"] == "unavailable"


def test_cloud_model_is_rejected_before_network(monkeypatch):
    monkeypatch.setenv("LOCAL_QWEN_MODEL", "qwen3:cloud")
    def forbidden(_):
        pytest.fail("Cloud model must not make a request")
    assert analyze(state(), httpx.MockTransport(forbidden)).enrichment["diagnostic_analysis"]["status"] == "unavailable"


def test_missing_model_is_explicitly_unavailable():
    client = httpx.MockTransport(lambda _: httpx.Response(200, json={"models": []}))
    assert analyze(state(), client).enrichment["diagnostic_analysis"]["status"] == "unavailable"


def test_local_mode_uses_model_node_even_at_low_risk(monkeypatch):
    monkeypatch.setenv("INCIDENT_ANALYSIS_MODE", "ollama")
    assert GenericIncidentGraph._route_after_evidence(SimpleNamespace(risk_score=0)) == "llm_reasoning"


def test_real_graph_calls_parser_then_local_analyzer(monkeypatch):
    monkeypatch.setenv("INCIDENT_ANALYSIS_MODE", "ollama")
    from app.services import local_analysis
    monkeypatch.setattr(local_analysis, "reasoning_node", lambda s: analyze(s, transport()))
    monkeypatch.setattr("app.services.generic_incident_graph.reasoning_node", local_analysis.reasoning_node)
    report = GenericIncidentGraph().investigate(state().incident)
    assert report.diagnostic_analysis["status"] == "completed"
    assert report.diagnostic_analysis["parsed_evidence"]["request_groups"]


def test_api_uses_validated_summary_without_sending_actions(monkeypatch):
    from app.services import local_analysis
    report = {"executive_summary": "offline", "risk_score": 10, "investigation_id": "demo",
              "references": [], "evidence": [], "recommended_actions": [],
              "diagnostic_analysis": analyze(state(), transport()).enrichment["diagnostic_analysis"]}
    monkeypatch.setattr(api, "investigate", lambda *_: report)
    result = api.execute(api.Request(question="Checkout reports HTTP 503"))
    assert result["mode"] == "local_qwen"
    assert "not confirmed" in result["answer"]
    assert result["ticket_draft"]["priority"] == "high"
    assert result["ticket_draft"]["next_action"] == "human_review_before_submission"
