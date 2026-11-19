import json
from pathlib import Path
from unittest.mock import patch

import api as c
from domain_classifier import predict
from evidence_provenance import annotate
from route_verification import verify_route

def test_ml_handles_unmatched_policy_language():
    for question,domain in [('Unexpected OAuth permissions on user identity','security')]:
        decision=c.route(c.Request(question=question))
        assert decision['domain']==domain
        assert decision['routing_mode']=='ml_fallback'
        assert decision['model_prediction']['accepted']

def test_missing_model_and_unknown_requests_clarify():
    with patch('routing_model.load_manifest',return_value=None), patch('domain_classifier.load_model',return_value=None):
        assert c.route(c.Request(question='Desktop web browser stopped opening pages'))['path']=='clarify'
    assert c.route(c.Request(question='CPU 98 percent please help'))['path']=='clarify'
    assert c.route(c.Request(question='Write a birthday poem'))['path']=='clarify'

def test_explicit_and_ambiguous_routes_do_not_depend_on_ml():
    with patch('domain_router.predict',side_effect=AssertionError('ML must not override explicit/tied domains')):
        assert c.route(c.Request(question='VPN failure',domain='cloud'))['domain']=='cloud'
        assert c.route(c.Request(question='ETL failure and Kubernetes failure'))['path']=='clarify'

def test_priority_policy_is_retrieved_with_vpn_topic():
    result=c.support('Multiple users cannot access VPN')
    assert 'vpn_troubleshooting_kb.md' in result['sources']
    assert 'priority_matrix.md' in result['sources']

def test_team_policy_is_retrieved_with_vpn_topic():
    result=c.support('Which support team handles VPN connectivity issues?')
    assert 'vpn_troubleshooting_kb.md' in result['sources']
    assert 'ticket_routing_rules.md' in result['sources']
    assert len(result['sources'])==len(set(result['sources']))

def test_evidence_origin_does_not_imply_verification():
    report=annotate({'evidence':[{'kind':'log','value':'user log'},{'kind':'mcp_tool','value':'simulated signal'},
                                {'kind':'access_impact','value':'simulated agent observation'},
                                {'kind':'triage_signal','value':'rule decision'}], 'findings':[{'summary':'assessment'}]})
    assert report['evidence_summary']=={'user_supplied':1,'local_simulation':2,'rule_derived':1,'independently_verified':0}
    assert all(not e['independently_verified'] for e in report['evidence'])
    assert report['semantic_groundedness']=='not_evaluated'
    assert report['findings'][0]['verification_status']=='unverified_offline_assessment'

def test_dataset_has_disjoint_text_splits():
    rows=json.loads(Path('domain_dataset.json').read_text())['examples']
    assert len(rows)==len({r['text'].strip().lower() for r in rows})
    assert {r['split'] for r in rows}=={'train','validation','test'}


def test_explicit_vpn_alias_and_missing_context_policy():
    decision=c.route(c.Request(question='My virtual private network stopped connecting'))
    assert decision['domain']=='it' and decision['routing_mode']=='weighted_policy'
    assert c.route(c.Request(question='Something is broken'))['path']=='clarify'


def test_http_server_errors_are_explicit_production_signals():
    decision=c.route(c.Request(question='Ingress reports HTTP 502 errors'))
    assert decision['domain']=='production' and decision['routing_mode']=='weighted_policy'


def investigation_report(domain, evidence=()):
    return {'executive_summary': f'{domain} findings', 'investigation_id': domain,
            'references': [], 'evidence': list(evidence), 'recommended_actions': [], 'risk_score': 40}


def test_supplied_evidence_can_reroute_once_and_preserve_initial_output(monkeypatch):
    request = c.Request(question='VPN stopped connecting', events=[{'event': 'identity.mfa_fatigue'}])
    assert c.route(request)['domain'] == 'it'
    calls = []

    def investigate(input_request, domain):
        assert input_request is request
        calls.append(domain)
        return investigation_report(domain)

    monkeypatch.setattr(c, 'investigate', investigate)
    monkeypatch.setattr(c, 'support', lambda _, **kwargs: {
        'answer': 'VPN guidance', 'ticket_draft': {'assigned_team': 'Network Support', 'priority': 'Low'},
    })
    result = c.execute(request)
    assert calls == ['it', 'security']
    assert result['initial_routing']['domain'] == 'it'
    assert result['routing']['domain'] == 'security'
    assert result['route_verification']['status'] == 'rerouted'
    assert result['previous_investigation']['investigation_id'] == 'it'
    assert result['investigation']['investigation_id'] == 'security'
    assert result['answer'] == 'security findings'
    assert result['support']['answer'] == 'VPN guidance'
    assert result['ticket_draft']['assigned_team'] == 'SOC'
    assert result['ticket_draft']['next_action'] == 'human_review_before_submission'


def test_simulation_and_retrieved_sources_cannot_reroute():
    request = c.Request(question='VPN stopped connecting')
    report = investigation_report('it', [
        {'kind': 'log', 'origin': 'local_simulation', 'value': 'identity.mfa_fatigue'},
        {'kind': 'log', 'origin': 'rule_derived', 'value': 'identity.account_takeover'},
    ])
    report['references'] = [{'source': 'identity.mfa_fatigue'}]
    assert verify_route(request, 'it', report)['status'] == 'unverified'


def test_explicit_domain_and_conflicting_telemetry_require_review():
    explicit = c.Request(question='VPN stopped connecting', domain='it',
                         events=[{'event': 'identity.mfa_fatigue'}])
    check = verify_route(explicit, 'it')
    assert check['status'] == 'review' and check['suggested_domain'] == 'security'
    conflicting = c.Request(question='VPN stopped connecting', logs=['vpn.certificate_expired device=a'],
                            events=[{'event': 'identity.mfa_fatigue'}])
    check = verify_route(conflicting, 'it')
    assert check['status'] == 'review' and check['suggested_domain'] is None


def test_matching_telemetry_supports_but_does_not_independently_verify_route():
    request = c.Request(question='VPN stopped connecting', logs=['event=vpn.certificate_expired device=a'])
    result = verify_route(request, 'it')
    assert result['status'] == 'supported'
    assert result['signals'] == {'it': ['vpn.certificate_expired']}


def test_second_workflow_disagreement_stops_for_review(monkeypatch):
    request = c.Request(question='Checkout returns HTTP 503', events=[{'event': 'cloud.quota_exceeded'}])
    calls = []

    def investigate(input_request, domain):
        calls.append(domain)
        evidence = [] if domain == 'production' else [
            {'kind': 'log', 'origin': 'user_supplied', 'value': 'service.healthcheck_failed'},
        ]
        return investigation_report(domain, evidence)

    monkeypatch.setattr(c, 'investigate', investigate)
    result = c.execute(request)
    assert calls == ['production', 'cloud']
    assert result['route_verification']['rerouted']
    assert result['route_verification']['status'] == 'review'
    assert result['route_verification']['review_required']
