"""One classification entry point; retain the existing trained models and policy."""
from domain_router import route


def classify_intake(request, ticket_classifier):
    routing = route(request)
    ticket = None
    if routing["path"] != "clarify" and (routing["domain"] == "it" or routing["support_context"]):
        ticket = ticket_classifier(request.question)
    return {
        "routing": routing,
        "domain": routing["domain"],
        "workflow_path": routing["path"],
        "subcategory": ticket["ticket"]["category"] if ticket else None,
        "priority": ticket["ticket"]["priority"] if ticket else request.severity,
        "suggested_team": ticket["ticket"]["assigned_team"] if ticket else None,
        "support_classification": ticket,
        "method": ("qwen_qlora_primary_plus_existing_it_ticket_model"
                   if routing.get("routing_mode") == "qwen_qlora_primary"
                   else "existing_domain_policy_and_models_plus_existing_it_ticket_model"),
        "scope": "initial_intake",
    }
