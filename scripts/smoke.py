"""Exercise real HTTP and save ids for verification after a Compose restart."""
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'http://127.0.0.1:8000'
STATE = Path('/data/smoke.json')


def request(method, path, body=None, expected=200):
    data = json.dumps(body).encode() if body is not None else None
    req = Request(BASE + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        response = urlopen(req, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        result = json.load(response)
        assert response.code == expected, (path, response.code, result)
        return result


def search(q):
    return request('GET', '/search?' + urlencode({'q': q}))


def verify(state):
    assert any(r['id'] == state['client']['id'] for r in search('NevisWealth'))
    assert any(r['id'] == state['client']['id'] and r['type'] == 'client'
               and r['match_type'] == 'literal' for r in search('John NevisWealth'))
    matches = search('address proof')
    assert any(r['id'] == state['document']['id'] and r['match_type'] == 'semantic' for r in matches)
    assert not any(r['id'] == state['unrelated']['id'] for r in matches)
    assert any(r['id'] == state['missing']['id'] and r['match_type'] == 'semantic' for r in matches)
    assert not any(r['id'] == state['algorithm']['id'] for r in matches)
    assert search('quantum entanglement') == []
    assert any(r['id'] == state['document']['id'] for r in search('utility bill'))
    assert request('GET', '/openapi.json')['openapi'].startswith('3.1.')


def main():
    if sys.argv[1:] == ['--verify-restart']:
        verify(json.loads(STATE.read_text()))
        print('Restart persistence: PASS')
        return
    assert not sys.argv[1:], 'Usage: smoke.py [--verify-restart]'
    payload = {
        'first_name': 'John', 'last_name': 'Doe', 'email': 'john.doe@neviswealth.com',
        'description': 'Advisor based in London', 'social_links': ['@john'],
    }
    existing = [item for item in search(payload['email'])
                if item['type'] == 'client' and item['email'] == payload['email']]
    client = ({key: value for key, value in existing[0].items() if key not in ('type', 'match_type')}
              if existing else request('POST', '/clients', payload, 201))
    request('POST', '/clients', {**payload, 'email': 'JOHN.DOE@NEVISWEALTH.COM'}, 409)
    document = request('POST', f"/clients/{client['id']}/documents", {
        'title': 'Residence record', 'content': 'utility bill',
    }, 201)
    unrelated = request('POST', f"/clients/{client['id']}/documents", {
        'title': 'Cooking',
        'content': 'Chocolate cake recipe: combine flour, eggs and cocoa; bake for thirty minutes.',
    }, 201)
    missing = request('POST', f"/clients/{client['id']}/documents", {
        'title': 'Outstanding residence record', 'content': 'The client has not provided a utility bill yet.',
    }, 201)
    algorithm = request('POST', f"/clients/{client['id']}/documents", {
        'title': 'Algorithm note', 'content': 'We need to prove that an algorithm is correct.',
    }, 201)
    state = {'client': client, 'document': document, 'unrelated': unrelated, 'missing': missing, 'algorithm': algorithm}
    verify(state)
    request('POST', '/clients/nonexistent/documents', {'title': 'Missing', 'content': 'Example'}, 404)
    request('POST', '/clients', {'first_name': 'Invalid'}, 422)
    request('GET', '/search?q=%20%20', expected=422)
    request('GET', '/search', expected=422)
    STATE.write_text(json.dumps(state, indent=2))
    print(json.dumps({'client': client, 'created_document': document,
                      'email_search': search('NevisWealth'),
                      'semantic_search': search('address proof')}, indent=2))
    print('HTTP smoke: PASS')


if __name__ == '__main__':
    main()
