"""Check the provisional domain against narrowly scoped, supplied telemetry."""
import re


# Match event codes, not words in a generated answer or retrieved runbook.
SIGNALS = {
    "security": {"identity.mfa_fatigue", "identity.account_takeover", "identity.password_spray"},
    "it": {"vpn.certificate_expired", "vpn.tunnel_negotiation_failed"},
    "production": {"service.healthcheck_failed", "service.http_5xx_threshold_breached"},
    "cloud": {"cloud.quota_exceeded", "cloud.node_capacity_exhausted"},
    "data": {"data.schema_mismatch", "data.pipeline_integrity_failed"},
}
LOG_CODE = re.compile(r"^(?:event=)?([a-z][a-z0-9_.]+)(?:\s|$)", re.IGNORECASE)


def verify_route(request, domain, report=None):
    """Return an advisory result; supplied telemetry is not independently verified."""
    codes = {
        event["event"].strip().lower()
        for event in [*request.events, *request.raw_events]
        if isinstance(event.get("event"), str)
    }
    logs = list(request.logs)
    for item in (report or {}).get("evidence", []):
        if item.get("origin") == "user_supplied" and item.get("kind") == "log":
            value = item.get("value")
            if isinstance(value, str):
                logs.append(value)
    for line in logs:
        match = LOG_CODE.match(line.strip())
        if match:
            codes.add(match[1].lower())

    signals = {name: sorted(codes & known) for name, known in SIGNALS.items() if codes & known}
    check = {"signals": signals, "suggested_domain": None, "review_required": True}
    if not signals:
        return {**check, "status": "unverified", "reason": "No recognized user-supplied telemetry confirmed the route."}
    if len(signals) > 1:
        return {**check, "status": "review", "reason": "User-supplied telemetry points to multiple domains."}
    candidate = next(iter(signals))
    if candidate == domain:
        return {**check, "status": "supported", "review_required": False,
                "reason": "Recognized user-supplied telemetry supports the selected domain."}
    if request.domain != "auto":
        return {**check, "status": "review", "suggested_domain": candidate,
                "reason": "Telemetry disagrees with the explicitly selected domain; no automatic reroute."}
    return {**check, "status": "alternate", "suggested_domain": candidate,
            "reason": "Recognized user-supplied telemetry points to a different domain."}
