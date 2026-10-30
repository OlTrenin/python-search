"""Mandatory acceptance checks against the real pinned model, never mocked vectors."""
import numpy as np
import pytest
import json
from pathlib import Path

from nevis.embeddings import Embeddings, SIMILARITY_THRESHOLD
from nevis.reranker import RERANK_THRESHOLD

pytestmark = pytest.mark.semantic


@pytest.fixture(scope='module')
def model():
    return Embeddings()


def test_pdf_semantic_example(model):
    query = np.array(model.query('address proof'))
    vectors = model.document('utility bill')
    assert max(np.dot(query, vector) for vector in vectors) >= SIMILARITY_THRESHOLD


@pytest.mark.parametrize(('query', 'content'), [
    ('address proof', "A recent utility bill showing the client's residential address."),
    ('identity verification', "A copy of the client's passport photograph page."),
    ('employment income', 'Monthly payslip showing salary and employer details.'),
])
def test_related_concepts(model, query, content):
    vector = np.array(model.query(query))
    assert max(np.dot(vector, chunk) for chunk in model.document(content)) >= SIMILARITY_THRESHOLD


@pytest.mark.parametrize(('query', 'content'), [
    ('address proof', 'Chocolate cake recipe: combine flour, eggs and cocoa; bake for thirty minutes.'),
    ('quantum entanglement', 'utility bill'),
    ('quantum entanglement', "A copy of the client's passport photograph page."),
    ('quantum entanglement', 'Monthly payslip showing salary and employer details.'),
])
def test_unrelated_concepts_rejected(model, query, content):
    assert list(model.rerank(query, [content]))[0] < RERANK_THRESHOLD


def test_semantic_match_beyond_first_context_window(model):
    content = 'Chocolate cake recipe. Mix flour and cocoa then bake. ' * 100
    content += "\nA recent utility bill showing the client's residential address."
    vector = np.array(model.query('address proof'))
    chunks = model.document(content)
    assert list(model.rerank('address proof', [content.splitlines()[0]]))[0] < RERANK_THRESHOLD
    assert max(np.dot(vector, chunk) for chunk in chunks) >= SIMILARITY_THRESHOLD


def test_chunk_budget_without_redundant_short_tail(model):
    assert len(model.document('bill ' * 250)) == 1
    for vector in model.document('bill ' * 600):
        assert len(vector) == 384
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(ValueError, match='token limit'):
        model.query('bill ' * 511)


def test_real_model_through_unified_http_search(model, tmp_path):
    from fastapi.testclient import TestClient
    from nevis.app import create_app

    with TestClient(create_app(tmp_path / 'acceptance.sqlite3', model)) as client:
        owner = client.post('/clients', json={
            'first_name': 'John', 'last_name': 'Doe', 'email': 'john.doe@neviswealth.com',
        }).json()
        cases = [
            ('address proof', 'utility bill'),
            ('identity verification', "A copy of the client's passport photograph page."),
            ('employment income', 'Monthly payslip showing salary and employer details.'),
            ('address proof', 'Cake instructions. ' * 1000 + '\nA recent utility bill showing the residential address.'),
        ]
        documents = []
        for query, content in cases:
            response = client.post(f"/clients/{owner['id']}/documents", json={'title': 'Record', 'content': content})
            assert response.status_code == 201
            documents.append((query, response.json()))
        cake = client.post(f"/clients/{owner['id']}/documents", json={
            'title': 'Cooking', 'content': 'Chocolate cake recipe: mix flour and cocoa then bake.',
        }).json()
        assert any(item['id'] == owner['id'] for item in client.get('/search', params={'q': 'NevisWealth'}).json())
        for query, document in documents:
            response = client.get('/search', params={'q': query})
            assert response.status_code == 200
            matches = response.json()
            assert any(item['id'] == document['id'] and item['match_type'] == 'semantic' for item in matches)
            assert not any(item['id'] == cake['id'] for item in matches)
            assert matches == client.get('/search', params={'q': query}).json()
        assert client.get('/search', params={'q': 'quantum entanglement'}).json() == []
        fragmented = client.post(f"/clients/{owner['id']}/documents", json={
            'title': 'Fragmentation check', 'content': 'bill\n' * 513,
        })
        assert fragmented.status_code == 422
        assert fragmented.json()['detail'][0]['loc'] == ['body', 'content']
        assert not any(item.get('title') == 'Fragmentation check' for item in
                       client.get('/search', params={'q': 'Fragmentation check'}).json())


def test_rejects_excessive_fragmentation_before_inference(model):
    with pytest.raises(ValueError, match='512 semantic chunks'):
        model.document('bill\n' * 513)


def test_address_proof_returns_bill_and_missing_bill_but_not_algorithm(model, tmp_path):
    from fastapi.testclient import TestClient
    from nevis.app import create_app

    with TestClient(create_app(tmp_path / 'proof.sqlite3', model)) as api:
        owner = api.post('/clients', json={
            'first_name': 'Ada', 'last_name': 'Lovelace', 'email': 'ada@example.com',
        }).json()
        ids = []
        for content in [
            'utility bill',
            'The client has not provided a utility bill yet.',
            'We need to prove that an algorithm is correct.',
        ]:
            response = api.post(f"/clients/{owner['id']}/documents", json={
                'title': 'Record', 'content': content,
            })
            assert response.status_code == 201
            ids.append(response.json()['id'])
        response = api.get('/search', params={'q': 'address proof'})
        assert response.status_code == 200
        assert {item['id'] for item in response.json()} == set(ids[:2])
        assert all(item['match_type'] == 'semantic' for item in response.json())


@pytest.mark.parametrize('split', ['calibration', 'validation', 'holdout'])
def test_labeled_relevance_through_http(model, tmp_path, split):
    from fastapi.testclient import TestClient
    from nevis.app import create_app

    cases = json.loads(Path(__file__).with_name('relevance_cases.json').read_text())[split]
    for index, case in enumerate(cases):
        with TestClient(create_app(tmp_path / f'{split}-{index}.sqlite3', model)) as api:
            owner = api.post('/clients', json={
                'first_name': 'Ada', 'last_name': 'Lovelace', 'email': 'ada@example.com',
            }).json()
            expected = set()
            for relevant, contents in [(True, case['relevant']), (False, case['irrelevant'])]:
                for content in contents:
                    response = api.post(f"/clients/{owner['id']}/documents", json={
                        'title': 'Record', 'content': content,
                    })
                    assert response.status_code == 201
                    if relevant:
                        expected.add(response.json()['id'])
            response = api.get('/search', params={'q': case['q']})
            assert response.status_code == 200
            assert {item['id'] for item in response.json()} == expected, (split, case['q'], response.json())


def test_relevance_query_rejects_context_overflow(model):
    with pytest.raises(ValueError, match='relevance model token limit'):
        model.query('a ' * 249)
