"""Local integration prototype: support, investigation, and reviewed ticket drafts."""
import os
import secrets
import sqlite3
import sys
import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / 'incident'))
os.environ.update(LLM_PROVIDER='offline', OPENAI_API_KEY='', ANTHROPIC_API_KEY='', USE_POSTGRES='false',
                  LANGFUSE_ENABLED='false', A2A_PROVIDER='local', AUTH_REQUIRED='true', JWT_SECRET_KEY=secrets.token_hex(32))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from app.core.security import CurrentUser, create_access_token, get_current_user
from app.models.schemas import IncidentInput, SecurityAlert, UserRole
from app.services.generic_incident_graph import GenericIncidentGraph
from app.services.investigation_graph import InvestigationGraph
from domain_router import route
from platform_workflow import PlatformWorkflow
from app.rag.retriever import SecurityKnowledgeRetriever
from app.services.local_analysis import analyze as analyze_local_evidence, redact
from types import SimpleNamespace
from copy import deepcopy
from evidence_provenance import annotate
from support_client import worker
from public_faq_cache import PublicFAQCache
from external_routes import build_router
from job_routes import build_job_router
from routing_model import embedding_worker as routing_worker

faq_cache = PublicFAQCache(ROOT / 'university_eval_questions.json')
CASES_DB = Path(os.getenv('CASES_DB_PATH', str(ROOT / 'cases.sqlite3')))

@asynccontextmanager
async def lifespan(app):
    get_job_queue()
    try:
        yield
    finally:
        worker.close()
        routing_worker.close()

api = FastAPI(title='Enterprise IT Support and Incident Investigation - Local Prototype', lifespan=lifespan)
TEAMS = {'security':'SOC', 'production':'SRE', 'cloud':'Cloud Infrastructure', 'data':'Data Engineering', 'it':'IT Operations'}

class Request(BaseModel):
    question: str = Field(min_length=3, max_length=12000)
    domain: Literal['auto','security','production','cloud','data','it'] = 'auto'
    severity: Literal['low','medium','high','critical'] = 'medium'
    logs: list[str] = Field(default_factory=list, max_length=200)
    events: list[dict] = Field(default_factory=list, max_length=200)
    raw_events: list[dict] = Field(default_factory=list, max_length=200)
    metrics: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list, max_length=100)
    id: str | None = None
    title: str | None = None
    description: str | None = None
    source: str = 'manual'
    service: str | None = None
    environment: str | None = None
    owner_team: str | None = None
    user: str | None = None
    host: str | None = None
    ip_address: str | None = None
    geo: str | None = None
    tactic: str | None = None
    technique: str | None = None
    detected_at: datetime | None = None

class Login(BaseModel):
    username: str
    password: str

def support(question, classification=None):
    if classification is not None:
        # An injected decision must not be replaced by an older cached ticket.
        return dict(worker.call(question, classification),
                    delivery={'public_faq_cache_hit': False, 'cache_eligible': False})
    return faq_cache.get(question, lambda: worker.call(question))

def investigate(request, domain):
    common = {'title': request.title or request.question[:150],
              'description': request.description if request.description is not None else request.question,
              'severity': request.severity, 'source': request.source, 'tags': request.tags}
    if request.id:
        common['id'] = request.id
    if request.detected_at:
        common['detected_at'] = request.detected_at
    if domain == 'security':
        payload = SecurityAlert(**common, user=request.user, host=request.host, ip_address=request.ip_address,
                                geo=request.geo, tactic=request.tactic, technique=request.technique,
                                raw_events=[*request.raw_events, *request.events,
                                            *[{'event':'submitted_log','message':line} for line in request.logs]])
        return annotate(InvestigationGraph().investigate(payload).model_dump(mode='json'))
    payload = IncidentInput(**common, domain=domain, service=request.service, environment=request.environment,
                            owner_team=request.owner_team, logs=request.logs, metrics=request.metrics,
                            events=[*request.events, *request.raw_events])
    return annotate(GenericIncidentGraph().investigate(payload).model_dump(mode='json'))

def add_support_result(result, question):
    intake = result['classification']
    if intake['support_classification'] is None:
        details = worker.classify(question)
        intake.update(support_classification=details, subcategory=details['ticket']['category'],
                      priority=details['ticket']['priority'], suggested_team=details['ticket']['assigned_team'])
    result['support'] = support(question, classification=intake['support_classification'])
    result['ticket_draft'] = dict(result['support']['ticket_draft'])
    result['answer'] = result['support']['answer']
    result['execution_trace'].append('LlamaIndex support workflow / NLP classifier / support RAG')


def add_investigation_trace(result, domain):
    result['execution_trace'].append(f'LangGraph {domain} investigation / local tools / runbook retrieval')


def retrieve_documentation(query, domain, include_support=False):
    query = redact(query)[:300]
    references = SecurityKnowledgeRetriever().search_domain(query, domain)
    runbooks = [{'text': reference.content, 'source': reference.source, 'corpus': 'incident_runbook'}
                for reference in references[:3]]
    return worker.retrieve(query, domain, runbooks, include_support)


def refine_analysis(request, report, documents):
    previous = report['diagnostic_analysis']
    severity = previous.get('effective_severity', request.severity)
    domain = report.get('incident', {}).get('domain', 'security' if 'alert' in report else 'production')
    state = SimpleNamespace(
        incident=IncidentInput(title=request.question[:150], description=request.question,
                               domain=domain, severity=severity),
        references=[], timeline=[],
        enrichment={'parsed_evidence': previous['parsed_evidence'], 'additional_documentation': documents})
    analyze_local_evidence(state)
    revised = deepcopy(report)
    revised['diagnostic_analysis'] = state.enrichment['diagnostic_analysis']
    if revised['diagnostic_analysis']['status'] == 'completed':
        revised['executive_summary'] = revised['diagnostic_analysis']['assessment']['summary']
    revised.setdefault('timeline', []).extend(state.timeline)
    return revised


def compose_result(request, result, report):
    domain = result['routing']['domain']
    if report is not None:
        result['investigation'] = report
        result['answer'] = report['executive_summary']
        draft = dict(result['ticket_draft'] or {})
        support_team = draft.get('assigned_team')
        investigation_sources = list(dict.fromkeys(r['source'] for r in report['references']))
        # Support citations and ownership remain distinct from investigation evidence.
        draft.update(title=request.title or request.question[:150], investigation_summary=report['executive_summary'],
                     investigation_id=report['investigation_id'], evidence=report['evidence'],
                     recommended_actions=report['recommended_actions'],
                     investigation_source_references=investigation_sources,
                     next_action='human_review_before_submission')
        draft.setdefault('description',report['executive_summary'])
        draft.setdefault('source_references',investigation_sources)
        draft.setdefault('category',domain)
        draft['support_team'] = support_team
        draft['assigned_team'] = support_team if domain == 'it' and support_team else TEAMS[domain]
        draft['assignment_reason'] = 'Preserved specialist IT team.' if domain == 'it' and support_team else f'{domain} investigation owner; prior support team retained separately.'
        priorities = ['low','medium','high','critical']
        original = priorities.index(str(draft.get('priority','low')).lower())
        explicit = priorities.index(request.severity) if 'severity' in request.model_fields_set else 0
        draft['priority'] = priorities[max(original, explicit, 2 if report['risk_score'] >= 75 else 0)]
        result['ticket_draft'] = draft
    analysis = (result.get('investigation') or {}).get('diagnostic_analysis', {})
    if analysis:
        result['mode'] = 'local_qwen' if analysis['status'] == 'completed' else 'local_qwen_unavailable'
        result['limitations'][1] = 'Local-only Qwen is requested. Simulated tool evidence is excluded from its prompt. No external actions are executed.'
        result['limitations'].append('Qwen output is advisory; evidence IDs and schema do not prove factual correctness.')
        if analysis['status'] == 'completed':
            result['answer'] = analysis['assessment']['summary']
            draft = result.get('ticket_draft')
            if draft:
                draft['diagnostic_analysis'] = analysis['assessment']
                draft['investigation_summary'] = result['answer']
                ranks = ['low', 'medium', 'high', 'critical']
                draft['priority'] = ranks[max(ranks.index(draft['priority'].lower()),
                                              ranks.index(analysis['effective_severity']))]
        else:
            result['answer'] = analysis['reason'] + ' ' + result['answer']
    return result


def execute(request):
    return PlatformWorkflow(sys.modules[__name__]).run(request)

@api.get('/', include_in_schema=False)
def home():
    return FileResponse(ROOT/'support_dashboard.html')

@api.get('/architecture.svg', include_in_schema=False)
def architecture():
    return FileResponse(ROOT/'architecture.svg', media_type='image/svg+xml')

@api.post('/auth/login')
def login(credentials: Login):
    if not (secrets.compare_digest(credentials.username,'demo') and secrets.compare_digest(credentials.password,'local-demo-only')):
        raise HTTPException(401,'Invalid credentials')
    return {'access_token':create_access_token('demo',UserRole.admin),'token_type':'bearer'}

@api.post('/cases')
def create_case(request: Request, user: CurrentUser = Depends(get_current_user)):
    try:
        result = execute(request)
    except RuntimeError as error:
        raise HTTPException(503,'Workflow unavailable; no ticket was submitted.') from error
    with sqlite3.connect(CASES_DB, timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT)')
        db.execute('INSERT INTO cases VALUES (?, ?, ?)',(result['id'],user.username,json.dumps(result)))
    return result

@api.get('/cases')
def history(user: CurrentUser = Depends(get_current_user)):
    with sqlite3.connect(CASES_DB, timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, owner TEXT, result TEXT)')
        return [json.loads(row[0]) for row in db.execute('SELECT result FROM cases WHERE owner = ? ORDER BY rowid DESC LIMIT 20',(user.username,))]


api.include_router(build_router(lambda: CASES_DB, get_current_user))


job_router, get_job_queue = build_job_router(lambda: CASES_DB, get_current_user, Request)
api.include_router(job_router)
