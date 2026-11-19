"""LangGraph consumes preclassified input and coordinates support and investigation."""
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from langgraph.graph import StateGraph, START, END

from route_verification import verify_route


class PlatformState(TypedDict):
    request: object
    classification: dict
    result: dict
    report: object
    reroutes: int
    documents: list
    started: float


class PlatformWorkflow:
    def __init__(self, services):
        self.services = services
        graph = StateGraph(PlatformState)
        for name in ("initialize", "support", "investigate", "verify", "reroute",
                     "retrieve_documentation", "refine_analysis", "finalize"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "initialize")
        graph.add_conditional_edges("initialize", self.after_initialization,
                                    {"support": "support", "investigate": "investigate", "finalize": "finalize"})
        graph.add_conditional_edges("support", self.after_support,
                                    {"investigate": "investigate", "verify": "verify"})
        graph.add_edge("investigate", "verify")
        graph.add_conditional_edges("verify", self.after_verification,
                                    {"reroute": "reroute", "retrieve_documentation": "retrieve_documentation",
                                     "finalize": "finalize"})
        graph.add_conditional_edges("reroute", self.after_reroute,
                                    {"support": "support", "investigate": "investigate"})
        graph.add_conditional_edges("retrieve_documentation",
                                    lambda s: "refine_analysis" if s["documents"] else "finalize",
                                    {"refine_analysis": "refine_analysis", "finalize": "finalize"})
        graph.add_edge("refine_analysis", "finalize")
        graph.add_edge("finalize", END)
        self.graph = graph.compile()

    def run(self, request, classification, started=None):
        return self.graph.invoke({"request": request, "classification": classification,
                                  "result": {}, "report": None, "reroutes": 0, "documents": [],
                                  "started": perf_counter() if started is None else started})["result"]

    def initialize(self, state):
        request = state["request"]
        classification = state["classification"]
        result = {"id": str(uuid4()), "question": request.question, "classification": classification,
                  "routing": classification["routing"], "support": None, "investigation": None,
                  "ticket_draft": None,
                  "execution_trace": ["Unified classification before LangGraph", "Parent LangGraph / initialize"],
                  "orchestration_engine": "langgraph", "mode": "local_offline",
                  "limitations": ["Domain policy/model scores are uncalibrated.",
                                 "Local tools can be simulated; no external actions are executed.",
                                 "A human must review findings and ticket submission."]}
        if result["routing"]["path"] == "clarify":
            result.update(answer="Please identify the affected service or domain and describe the observed failure, impact, and available evidence.",
                          route_verification={"status": "not_run", "review_required": True,
                                              "reason": "Initial routing requires clarification."})
        return {"result": result}

    @staticmethod
    def after_initialization(state):
        routing = state["result"]["routing"]
        if routing["path"] == "clarify":
            return "finalize"
        if routing["domain"] == "it" or routing["support_context"]:
            return "support"
        return "investigate"

    def support(self, state):
        self.services.add_support_result(state["result"], state["request"].question)
        return {"result": state["result"]}

    @staticmethod
    def after_support(state):
        return "investigate" if state["result"]["routing"]["path"] == "investigate" else "verify"

    def investigate(self, state):
        domain = state["result"]["routing"]["domain"]
        report = self.services.investigate(state["request"], domain)
        self.services.add_investigation_trace(state["result"], domain)
        return {"report": report, "result": state["result"]}

    def verify(self, state):
        result = state["result"]
        check = verify_route(state["request"], result["routing"]["domain"], state["report"])
        if state["reroutes"]:
            if check["status"] == "alternate":
                check = {**check, "status": "review", "reason": "Routing still disputed; one-reroute limit reached."}
            check.update(rerouted=True, initial_check=result["route_verification"])
            if check["status"] == "supported":
                check["status"] = "rerouted"
        result["route_verification"] = check
        return {"result": result}

    @staticmethod
    def documentation_query(state):
        report = state["report"] or {}
        analysis = report.get("diagnostic_analysis", {})
        if analysis.get("status") != "completed":
            return ""
        return analysis.get("assessment", {}).get("documentation_query", "").strip()

    def after_verification(self, state):
        if state["result"]["route_verification"]["status"] == "alternate" and not state["reroutes"]:
            return "reroute"
        if self.documentation_query(state):
            return "retrieve_documentation"
        return "finalize"

    def reroute(self, state):
        result = state["result"]
        previous = result["routing"]
        domain = result["route_verification"]["suggested_domain"]
        result.update(initial_routing=previous, previous_investigation=state["report"])
        result["routing"] = {"domain": domain, "path": "investigate", "routing_mode": "evidence_reroute",
                             "support_context": bool(result["support"]) or domain == "it",
                             "reason": result["route_verification"]["reason"]}
        result["execution_trace"].append(f"Route verification: {previous['domain']} -> {domain} (one reroute allowed)")
        return {"result": result, "report": None, "reroutes": 1}

    @staticmethod
    def after_reroute(state):
        result = state["result"]
        return "support" if result["routing"]["domain"] == "it" and result["support"] is None else "investigate"

    def retrieve_documentation(self, state):
        result = state["result"]
        query = self.documentation_query(state)
        exchange = {"query": query, "rounds": 1, "max_rounds": 1,
                    "requested_by": "qwen", "status": "unavailable"}
        documents = []
        try:
            response = self.services.retrieve_documentation(query, result["routing"]["domain"],
                                                            bool(result["support"]))
            documents = response["documents"]
            exchange.update(status="retrieved" if documents else "no_results",
                            documents=documents, workflow_engine=response["workflow_engine"])
        except RuntimeError:
            exchange["reason"] = "Documentation lookup unavailable; initial assessment preserved."
        result["documentation_exchange"] = exchange
        result["execution_trace"].append("Parent LangGraph -> LlamaIndex documentation retrieval (one round)")
        return {"documents": documents, "result": result}

    def refine_analysis(self, state):
        result = state["result"]
        revised = self.services.refine_analysis(state["request"], state["report"], state["documents"])
        result["documentation_exchange"]["analysis_status"] = revised["diagnostic_analysis"]["status"]
        result["documentation_exchange"]["budget_exhausted"] = bool(
            revised["diagnostic_analysis"].get("assessment", {}).get("documentation_query", ""))
        if revised["diagnostic_analysis"]["status"] == "completed":
            result["initial_assessment"] = state["report"]["diagnostic_analysis"]
            report = revised
        else:
            report = state["report"]
            result["documentation_exchange"]["reason"] = "Follow-up model output unavailable; initial assessment preserved."
        result["execution_trace"].append("LlamaIndex documents -> Qwen reassessment; retrieval budget exhausted")
        return {"report": report, "result": result}

    def finalize(self, state):
        result = self.services.compose_result(state["request"], state["result"], state["report"])
        result["latency_ms"] = round((perf_counter() - state["started"]) * 1000, 2)
        return {"result": result}
