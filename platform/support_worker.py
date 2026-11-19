"""Persistent JSON-lines worker using the support project's environment."""
import contextlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).parent / 'support'
os.chdir(root)
sys.path.insert(0, str(root))
with contextlib.redirect_stdout(sys.stderr):
    from src.orchestrator import run_support_workflow
    from src.ticket_classifier import classify_ticket, classify_for_retrieval
    from src.documentation_workflow import retrieve_documentation

for line in sys.stdin:
    try:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            if request.get('operation', 'support') == 'classify':
                ticket = classify_ticket(request['question'])
                result = {'ticket': ticket, 'preliminary_classification': classify_for_retrieval(request['question'], ticket)}
            elif request.get('operation', 'support') == 'support':
                result = run_support_workflow(request['question'], classification=request.get('classification'))
            elif request.get('operation') == 'retrieve':
                if request['domain'] not in {'it', 'security', 'production', 'cloud', 'data'}:
                    raise ValueError('Unknown retrieval domain')
                result = retrieve_documentation(request['query'][:300], request['domain'],
                                                request['runbooks'], request.get('include_support', False))
            else:
                raise ValueError('Unknown support worker operation')
        response = {'ok': True, 'result': result}
    except Exception as error:
        print(f'Support request failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        response = {'ok': False, 'error': 'Support workflow failed'}
    print(json.dumps(response), flush=True)
