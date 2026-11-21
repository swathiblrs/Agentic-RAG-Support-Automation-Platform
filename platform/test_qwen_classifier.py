from unittest.mock import Mock

import api
import domain_router
from intake_classifier import classify_intake
from qwen_classifier import QwenClassifier


def prediction(domain='cloud', accepted=True, available=True):
    return {'domain': domain, 'accepted': accepted, 'available': available,
            'model': 'Qwen3-0.6B-QLoRA-v2', 'score': .97}


def test_qwen_is_primary_even_with_nonsecurity_policy_match(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    qwen = Mock(return_value=prediction())
    monkeypatch.setattr(domain_router, 'predict_qwen', qwen)
    monkeypatch.setattr(domain_router, 'predict', Mock(side_effect=AssertionError('No MiniLM')))
    result = domain_router.route(api.Request(question='VPN connection fails after infrastructure changes'))
    assert result['domain'] == 'cloud'
    assert result['routing_mode'] == 'qwen_qlora_primary'
    assert result['support_context'] and not result['classifier_fallback_used']
    qwen.assert_called_once()


def test_security_policy_can_override_primary_prediction(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', lambda text: prediction('it'))
    result = domain_router.route(api.Request(question='Suspicious login and account takeover'))
    assert result['domain'] == 'security' and result['routing_mode'] == 'security_policy'
    assert result['primary_prediction']['domain'] == 'it'


def test_explicit_domain_bypasses_qwen(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', Mock(side_effect=AssertionError('Explicit input')))
    result = domain_router.route(api.Request(question='VPN unavailable', domain='data'))
    assert result['domain'] == 'data' and result['routing_mode'] == 'explicit'


def test_uncertain_qwen_abstains_without_minilm_override(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', lambda text: prediction('it', accepted=False))
    monkeypatch.setattr(domain_router, 'predict', Mock(side_effect=AssertionError('No uncertainty override')))
    result = domain_router.route(api.Request(question='VPN has strange intermittent behavior'))
    assert result['path'] == 'clarify' and result['routing_mode'] == 'qwen_clarification'


def test_other_label_requests_clarification(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', lambda text: prediction('other'))
    assert domain_router.route(api.Request(question='Write a poem about a VPN'))['path'] == 'clarify'


def test_unavailable_qwen_reports_policy_fallback(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', lambda text: prediction(available=False))
    result = domain_router.route(api.Request(question='VPN stopped connecting'))
    assert result['domain'] == 'it' and result['routing_mode'] == 'weighted_policy'
    assert result['classifier_fallback_used'] and not result['primary_prediction']['available']


def test_legacy_mode_does_not_load_qwen(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'minilm')
    monkeypatch.setattr(domain_router, 'predict_qwen', Mock(side_effect=AssertionError('Qwen disabled')))
    result = domain_router.route(api.Request(question='VPN stopped connecting'))
    assert result['domain'] == 'it' and result['primary_classifier'] == 'minilm'


def test_qwen_decision_reused_by_unified_intake(monkeypatch):
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'qwen_qlora')
    monkeypatch.setattr(domain_router, 'predict_qwen', lambda text: prediction('it'))
    ticket = {'ticket': {'category': 'VPN', 'priority': 'Low', 'assigned_team': 'Network Support'}}
    classify = Mock(return_value=ticket)
    result = classify_intake(api.Request(question='VPN is unavailable'), classify)
    assert result['domain'] == 'it'
    assert result['method'] == 'qwen_qlora_primary_plus_existing_it_ticket_model'
    assert result['support_classification'] is ticket
    classify.assert_called_once()


def test_missing_files_never_start_or_download(monkeypatch, tmp_path):
    monkeypatch.setenv('QWEN_CLASSIFIER_DIR', str(tmp_path))
    monkeypatch.setattr('qwen_classifier.subprocess.Popen', Mock(side_effect=AssertionError('No downloader')))
    result = QwenClassifier().predict('VPN is unavailable')
    assert not result['available'] and 'no automatic download' in result['reason']


def test_invalid_configuration_rejected(monkeypatch):
    import pytest
    monkeypatch.setenv('DOMAIN_CLASSIFIER', 'misspelled')
    with pytest.raises(ValueError, match='DOMAIN_CLASSIFIER'):
        domain_router.route(api.Request(question='VPN is unavailable'))
