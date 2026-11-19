"""LlamaIndex documentation subworkflow called by the parent LangGraph."""
import asyncio

from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from src.retriever import retrieve_relevant_chunks


class DocumentsReady(Event):
    documents: list


class DocumentationWorkflow(Workflow):
    @step
    async def retrieve(self, ev: StartEvent) -> DocumentsReady:
        # Runbooks are scoped by the parent using its existing incident retriever.
        documents = list(ev.runbooks)[:3]
        if ev.domain == "it" or ev.include_support:
            documents.extend({**item, "corpus": "support"} for item in
                             retrieve_relevant_chunks(ev.query, top_k=2, domain="it_support"))
        return DocumentsReady(documents=documents)

    @step
    async def finish(self, ev: DocumentsReady) -> StopEvent:
        unique = {}
        for item in ev.documents:
            key = (item["source"], item["text"][:120])
            unique.setdefault(key, item)
        return StopEvent(result={"documents": list(unique.values())[:5],
                                 "workflow_engine": "llamaindex_documentation_workflow"})


def retrieve_documentation(query, domain, runbooks, include_support=False):
    async def run():
        workflow = DocumentationWorkflow(timeout=30)
        return await workflow.run(query=query, domain=domain, runbooks=runbooks, include_support=include_support)
    return asyncio.run(run())
