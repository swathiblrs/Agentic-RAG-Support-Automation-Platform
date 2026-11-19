from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import IncidentInput, RetrievedDocument, SecurityAlert
from app.rag.vector_store import LocalVectorStore
from app.storage.pgvector_store import PgVectorStore


class SecurityKnowledgeRetriever:
    DOMAIN_TAGS = {
        "production": {"production", "sre"}, "cloud": {"cloud", "infrastructure"},
        "data": {"data", "pipeline"}, "it": {"it", "vpn", "email", "endpoint"},
        "security": {"account-takeover", "siem", "T1078", "token-theft", "containment"},
    }

    def __init__(self, knowledge_dir: Path | None = None):
        base_dir = Path(__file__).resolve().parents[1]
        settings = get_settings()
        self.knowledge_dir = knowledge_dir or base_dir / "data" / "knowledge_base"
        self.store = LocalVectorStore(self.knowledge_dir)
        self.pgvector_store = PgVectorStore(settings.database_url) if settings.use_postgres else None

    def retrieve_for_alert(self, alert: SecurityAlert) -> list[RetrievedDocument]:
        settings = get_settings()
        query = " ".join(
            part
            for part in [
                alert.title,
                alert.description,
                alert.user or "",
                alert.host or "",
                alert.ip_address or "",
                alert.geo or "",
                alert.tactic or "",
                alert.technique or "",
                " ".join(alert.tags),
                " ".join(str(event) for event in alert.raw_events),
            ]
            if part
        )
        if self.pgvector_store is not None:
            try:
                return self.pgvector_store.search(query=query, top_k=settings.rag_top_k)
            except Exception:
                pass
        return self.search_domain(query, "security")

    def retrieve_for_incident(self, incident: IncidentInput) -> list[RetrievedDocument]:
        settings = get_settings()
        query = " ".join(
            part
            for part in [
                incident.title,
                incident.domain.value,
                incident.description,
                incident.service or "",
                incident.environment or "",
                incident.owner_team or "",
                " ".join(incident.tags),
                " ".join(incident.logs),
                " ".join(str(event) for event in incident.events),
                " ".join(f"{key}:{value}" for key, value in incident.metrics.items()),
            ]
            if part
        )
        if self.pgvector_store is not None:
            try:
                return self.pgvector_store.search(query=query, top_k=settings.rag_top_k)
            except Exception:
                pass
        return self.search_domain(query, incident.domain.value)

    def search_domain(self, query: str, domain: str) -> list[RetrievedDocument]:
        allowed = self.DOMAIN_TAGS[domain]
        candidates = self.store.search(query=query + " " + domain,
                                       top_k=max(len(self.store.documents), get_settings().rag_top_k))
        return [item for item in candidates if set(item.tags) & allowed][:get_settings().rag_top_k]
