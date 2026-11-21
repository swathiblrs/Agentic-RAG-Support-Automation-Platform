from unittest.mock import Mock
import api


def result(status):
    return {'route_verification': {'status': status}, 'execution_trace': [],
            'ticket_draft': {'next_action': 'draft'}, 'answer': 'Findings'}


def test_full_retry_reclassifies_once_and_flags_remaining_mismatch(monkeypatch):
    classify = Mock(return_value={'domain': 'it'})
    graph = Mock()
    graph.run.side_effect = [result('alternate'), result('alternate')]
    monkeypatch.setattr(api, 'classify_intake', classify)
    monkeypatch.setattr(api, 'PlatformWorkflow', lambda services: graph)
    output = api.execute(api.Request(question='VPN failed'))
    assert classify.call_count == 2 and graph.run.call_count == 2
    assert output['previous_attempt']['answer'] == 'Findings'
    assert output['classification_mismatch']['review_required']
    assert output['verification_retry']['reruns'] == 1


def test_retry_can_resolve(monkeypatch):
    graph = Mock()
    graph.run.side_effect = [result('alternate'), result('supported')]
    monkeypatch.setattr(api, 'classify_intake', Mock(return_value={}))
    monkeypatch.setattr(api, 'PlatformWorkflow', lambda services: graph)
    output = api.execute(api.Request(question='VPN failed'))
    assert output['verification_retry']['resolved']
    assert 'classification_mismatch' not in output


def test_missing_evidence_does_not_retry(monkeypatch):
    graph = Mock()
    graph.run.return_value = result('unverified')
    classify = Mock(return_value={})
    monkeypatch.setattr(api, 'classify_intake', classify)
    monkeypatch.setattr(api, 'PlatformWorkflow', lambda services: graph)
    output = api.execute(api.Request(question='VPN failed'))
    assert classify.call_count == graph.run.call_count == 1
    assert 'verification_retry' not in output
