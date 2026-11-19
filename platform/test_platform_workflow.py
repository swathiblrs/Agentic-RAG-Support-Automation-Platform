from types import SimpleNamespace
from unittest.mock import Mock

import api
from platform_workflow import PlatformWorkflow
import json
import httpx


def report(query=""):
    return {
        "executive_summary": "Initial assessment", "investigation_id": "demo",
        "risk_score": 50, "references": [], "evidence": [], "recommended_actions": [],
        "diagnostic_analysis": {"status": "completed", "effective_severity": "medium",
            "assessment": {"summary": "Initial assessment", "documentation_query": query},
            "parsed_evidence": {"records": [], "request_groups": {}}},
    }


def services():
    return SimpleNamespace(
        worker=SimpleNamespace(classify=Mock()),
        add_support_result=Mock(), investigate=Mock(return_value=report()),
        add_investigation_trace=Mock(), retrieve_documentation=Mock(),
        refine_analysis=Mock(), compose_result=lambda request, result, report: {**result, "investigation": report},
    )


def test_parent_graph_is_real_and_contains_support_and_classification():
    nodes = PlatformWorkflow(services()).graph.get_graph().nodes
    assert {"intake", "support", "investigate", "verify", "retrieve_documentation",
            "refine_analysis", "finalize"} <= set(nodes)


def test_production_without_documentation_gap_does_not_retrieve():
    deps = services()
    result = PlatformWorkflow(deps).run(api.Request(question="Checkout reports HTTP 503"))
    assert result["orchestration_engine"] == "langgraph"
    deps.investigate.assert_called_once()
    deps.retrieve_documentation.assert_not_called()
    deps.worker.classify.assert_not_called()


def test_qwen_requests_documents_then_receives_them_once():
    deps = services()
    deps.investigate.return_value = report("Timeout runbook for storage dependencies")
    docs = [{"source": "runbook.md", "text": "Check storage request timeouts."}]
    deps.retrieve_documentation.return_value = {"documents": docs, "workflow_engine": "llamaindex_documentation_workflow"}
    revised = report("Another query that must not loop")
    revised["executive_summary"] = "Revised assessment"
    deps.refine_analysis.return_value = revised
    result = PlatformWorkflow(deps).run(api.Request(question="Checkout reports HTTP 503"))
    deps.retrieve_documentation.assert_called_once_with("Timeout runbook for storage dependencies", "production", False)
    assert deps.refine_analysis.call_args.args[2] == docs
    assert result["investigation"]["executive_summary"] == "Revised assessment"
    assert result["documentation_exchange"]["rounds"] == 1
    assert result["documentation_exchange"]["budget_exhausted"]
    assert result["initial_assessment"]["assessment"]["summary"] == "Initial assessment"


def test_empty_or_failed_lookup_retains_original_assessment():
    for response in ({"documents": [], "workflow_engine": "llamaindex_documentation_workflow"}, RuntimeError()):
        deps = services()
        deps.investigate.return_value = report("Storage timeout guidance")
        if isinstance(response, Exception):
            deps.retrieve_documentation.side_effect = response
        else:
            deps.retrieve_documentation.return_value = response
        result = PlatformWorkflow(deps).run(api.Request(question="Checkout reports HTTP 503"))
        deps.refine_analysis.assert_not_called()
        assert result["investigation"]["executive_summary"] == "Initial assessment"
        assert result["documentation_exchange"]["status"] in {"no_results", "unavailable"}


def test_invalid_followup_retains_initial_analysis():
    deps = services()
    deps.investigate.return_value = report("Storage runbook")
    deps.retrieve_documentation.return_value = {"documents": [{"source": "guide.md", "text": "guidance"}],
                                              "workflow_engine": "llamaindex_documentation_workflow"}
    deps.refine_analysis.return_value = {"diagnostic_analysis": {"status": "unavailable"}}
    result = PlatformWorkflow(deps).run(api.Request(question="Checkout reports HTTP 503"))
    assert result["investigation"]["diagnostic_analysis"]["status"] == "completed"
    assert result["documentation_exchange"]["analysis_status"] == "unavailable"


def test_clarification_never_runs_support_model_or_investigation():
    deps = services()
    result = PlatformWorkflow(deps).run(api.Request(question="Something is broken"))
    assert result["routing"]["path"] == "clarify"
    deps.worker.classify.assert_not_called()
    deps.investigate.assert_not_called()
    deps.retrieve_documentation.assert_not_called()


def test_real_llamaindex_retrieval_subworkflow_returns_supplied_runbooks():
    docs = [{"source": "production.md", "text": "Review the failing service.", "corpus": "incident_runbook"}]
    result = api.worker.retrieve("Service timeout troubleshooting", "production", docs)
    assert result["workflow_engine"] == "llamaindex_documentation_workflow"
    assert result["documents"] == docs


def test_followup_passes_retrieved_text_to_local_qwen_with_original_evidence(monkeypatch):
    from test_local_analysis import state, answer
    from app.services.local_analysis import analyze
    parsed = state().enrichment["parsed_evidence"]
    initial = report()
    initial["diagnostic_analysis"]["parsed_evidence"] = parsed
    initial["diagnostic_analysis"]["effective_severity"] = "high"
    captured = []

    def handle(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:1.7b"}]})
        payload = json.loads(request.content)
        prompt = json.loads(payload["prompt"])
        assert payload["format"]["properties"]["severity_evidence_ids"]["items"]["enum"] == prompt["allowed_incident_evidence_ids"]
        assert payload["format"]["properties"]["documentation_query"]["enum"] == [""]
        captured.append(prompt)
        return httpx.Response(200, json={"done": True, "response": json.dumps(answer())})

    monkeypatch.setattr(api, "analyze_local_evidence", lambda s: analyze(s, httpx.MockTransport(handle)))
    revised = api.refine_analysis(api.Request(question="Checkout HTTP 503"), initial,
                                  [{"text": "Check storage retry policy.", "source": "storage.md"}])
    assert captured[0]["additional_documentation"][0]["text"] == "Check storage retry policy."
    assert captured[0]["evidence"][1]["id"] == "log-1"
    assert revised["diagnostic_analysis"]["effective_severity"] == "high"
    assert revised["executive_summary"] == answer()["summary"]
    assert initial["executive_summary"] == "Initial assessment"


def test_documentation_search_filters_out_unrelated_security_guides(monkeypatch):
    retrieve = Mock(return_value={"documents": [], "workflow_engine": "llamaindex_documentation_workflow"})
    monkeypatch.setattr(api.worker, "retrieve", retrieve)
    api.retrieve_documentation("storage request timeout retry procedure", "production")
    candidates = retrieve.call_args.args[2]
    assert candidates
    assert all(doc["source"] == "SRE Runbook PROD-001" for doc in candidates)


def test_initial_incident_retrieval_is_also_domain_scoped():
    from app.models.schemas import IncidentInput
    references = api.SecurityKnowledgeRetriever().retrieve_for_incident(
        IncidentInput(title="Storage request timeout after checkout failed", domain="production"))
    assert references
    assert all(item.source == "SRE Runbook PROD-001" for item in references)
