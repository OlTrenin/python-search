"""HTTP contracts with substitute vectors; semantic quality is tested separately."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from nevis.app import create_app


class FakeEmbeddings:
    def query(self, text):
        return [-1.0, 0.0]

    def document(self, content):
        return [[1.0, 0.0]]

    def rerank(self, query, contents):
        return [0.0] * len(contents)


@pytest.fixture
def api(tmp_path):
    with TestClient(create_app(tmp_path / "test.sqlite3", FakeEmbeddings())) as client:
        yield client


def test_create_client_defaults_unique_email_and_documentation(api):
    payload = {"first_name": " John ", "last_name": "Doe", "email": "john@neviswealth.com"}
    first = api.post("/clients", json=payload)
    assert first.status_code == 201
    body = first.json()
    UUID(body["id"])
    assert body == {**payload, "first_name": "John", "id": body["id"], "description": None, "social_links": []}
    second = api.post("/clients", json=payload)
    assert second.status_code == 409
    assert second.json() == {"detail": "A client with this email already exists"}
    assert api.get("/health").json() == {"status": "ok"}
    assert api.get("/docs").status_code == 200
    schema = api.get("/openapi.json").json()
    assert schema["paths"]["/clients"]["post"]["responses"]["201"]
    assert schema["paths"]["/clients"]["post"]["responses"]["409"]


@pytest.mark.parametrize("change", [
    {"first_name": None}, {"first_name": "   "}, {"last_name": 42}, {"email": "invalid"},
    {"email": None}, {"description": 7}, {"social_links": "@john"}, {"social_links": [7]},
    {"first_name": "x" * 201}, {"description": "x" * 10_001},
    {"social_links": ["x"] * 21}, {"social_links": ["x" * 2049]}, {"unknown": True},
])
def test_reject_invalid_client(api, change):
    payload = {"first_name": "John", "last_name": "Doe", "email": "john@example.com", **change}
    response = api.post("/clients", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]


@pytest.mark.parametrize("field", ["first_name", "last_name", "email"])
def test_reject_missing_client_field(api, field):
    payload = {"first_name": "John", "last_name": "Doe", "email": "john@example.com"}
    del payload[field]
    assert api.post("/clients", json=payload).status_code == 422


def test_create_document_association_timestamp_and_unknown_client(api):
    client = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"}).json()
    content = "  Water usage for September\n"
    before = datetime.now(timezone.utc)
    response = api.post(f"/clients/{client['id']}/documents", json={"title": " Utility bill ", "content": content})
    assert response.status_code == 201
    document = response.json()
    UUID(document["id"])
    assert document["client_id"] == client["id"]
    assert document["title"] == "Utility bill"
    assert document["content"] == content
    created = datetime.fromisoformat(document["created_at"].replace("Z", "+00:00"))
    assert before <= created <= datetime.now(timezone.utc)
    assert set(document) == {"id", "client_id", "title", "content", "created_at"}
    missing = api.post("/clients/unknown/documents", json={"title": "Utility bill", "content": "Content"})
    assert missing.status_code == 404
    assert missing.json() == {"detail": "Client not found"}


@pytest.mark.parametrize("payload", [
    {}, {"title": "Bill"}, {"content": "Text"}, {"title": None, "content": "Text"},
    {"title": " ", "content": "Text"}, {"title": 1, "content": "Text"},
    {"title": "Bill", "content": " \n "}, {"title": "Bill", "content": None},
    {"title": "Bill", "content": 4}, {"title": "x" * 501, "content": "Text"},
    {"title": "Bill", "content": "x" * 100_001}, {"title": "Bill", "content": "Text", "unknown": 1},
])
def test_invalid_document(api, payload):
    client = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"}).json()
    assert api.post(f"/clients/{client['id']}/documents", json=payload).status_code == 422


def test_search_substrings_unicode_punctuation_and_typed_array(api):
    payload = {"first_name": "Jörg", "last_name": "Straße", "email": "john.doe@neviswealth.com", "description": "Advisor for 50%_holdings and O'Reilly", "social_links": ["@joerg"]}
    client = api.post("/clients", json=payload).json()
    for query in ["JÖ", "STRASSE", "jÖrg sTRASSE", "NevisWealth", "VISOR", "%_", "O'Reilly", "  NevisWealth  "]:
        response = api.get("/search", params={"q": query})
        assert response.status_code == 200
        assert response.json() == [{**client, "type": "client", "match_type": "literal"}]
    assert api.get("/search", params={"q": "' OR 1=1 --"}).json() == []
    assert api.get("/search", params={"q": "quantum entanglement"}).json() == []
    document = api.post(f"/clients/{client['id']}/documents", json={"title": "Advisor registration", "content": "Other text"}).json()
    response = api.get("/search", params={"q": "visor"})
    assert response.json() == [{**client, "type": "client", "match_type": "literal"}, {**document, "type": "document", "match_type": "literal"}]
    assert api.get("/search", params={"q": "visor"}).json() == response.json()
    assert api.get("/search", params={"q": "OTHER TEXT"}).json() == [{**document, "type": "document", "match_type": "literal"}]
    schema = api.get("/openapi.json").json()
    result_schema = schema["paths"]["/search"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert result_schema["type"] == "array"
    assert result_schema["items"]["discriminator"]["propertyName"] == "type"


@pytest.mark.parametrize(('query', 'matches_john'), [
    ('John NevisWealth', True),
    ('  nEvIsWeAlTh \t JOHN  ', True),
    ('Doe John', True),
    ('JOHN LONDON', True),
    ('ohn evis', True),
    ('John Jane', False),
    ('John Paris', False),
    ('John Madrid', False),
    ('John missing', False),
])
def test_all_query_parts_match_one_client_across_fields(api, query, matches_john):
    john = api.post('/clients', json={
        'first_name': 'John', 'last_name': 'Doe', 'email': 'john.doe@neviswealth.com',
        'description': 'Advisor based in London', 'social_links': ['@paris'],
    }).json()
    assert api.post('/clients', json={
        'first_name': 'Jane', 'last_name': 'Doe', 'email': 'jane.doe@neviswealth.com',
        'description': 'Advisor based in Paris',
    }).status_code == 201
    assert api.post(f"/clients/{john['id']}/documents", json={
        'title': 'Record', 'content': 'John advises clients at NevisWealth in Madrid.',
    }).status_code == 201
    response = api.get('/search', params={'q': query})
    assert response.status_code == 200
    expected = [{**john, 'type': 'client', 'match_type': 'literal'}] if matches_john else []
    assert response.json() == expected


@pytest.mark.parametrize("params", [{}, {"q": ""}, {"q": " \t\n "}, {"q": "x" * 501}])
def test_invalid_query(api, params):
    response = api.get("/search", params=params)
    assert response.status_code == 422
    assert response.json()["detail"]


def test_semantic_result_contract_with_substitute_vectors(tmp_path):
    class MatchingEmbeddings(FakeEmbeddings):
        def query(self, text):
            return [1.0, 0.0]

    with TestClient(create_app(tmp_path / "test.sqlite3", MatchingEmbeddings())) as api:
        client = api.post("/clients", json={"first_name": "Needle", "last_name": "Client", "email": "n@example.com"}).json()
        document = api.post(f"/clients/{client['id']}/documents", json={"title": "Bill", "content": "Other content"}).json()
        results = api.get("/search", params={"q": "needle"}).json()
        assert results == [{**client, "type": "client", "match_type": "literal"}, {**document, "type": "document", "match_type": "semantic"}]
        assert api.get("/search", params={"q": "needle"}).json() == results


def test_embedding_failure_is_503_and_document_is_not_stored(tmp_path):
    class FailingEmbeddings(FakeEmbeddings):
        def document(self, content):
            raise RuntimeError("private backend detail")

    with TestClient(create_app(tmp_path / "test.sqlite3", FailingEmbeddings())) as api:
        client = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"}).json()
        payload = {"title": "Failed index", "content": "Never saved"}
        assert api.post("/clients/unknown/documents", json=payload).status_code == 404
        response = api.post(f"/clients/{client['id']}/documents", json=payload)
        assert response.status_code == 503
        assert response.json() == {"detail": "Semantic search temporarily unavailable"}
        assert api.get("/search", params={"q": "Never saved"}).json() == []


@pytest.mark.parametrize("error,status", [(ValueError("Query exceeds the semantic model token limit"), 422), (RuntimeError("private backend detail"), 503)])
def test_query_embedding_error(tmp_path, error, status):
    class FailingEmbeddings(FakeEmbeddings):
        def query(self, text):
            raise error

    with TestClient(create_app(tmp_path / "test.sqlite3", FailingEmbeddings())) as api:
        response = api.get("/search", params={"q": "some query"})
        assert response.status_code == status
        assert "private backend detail" not in response.text
        if status == 422:
            assert response.json()["detail"][0]["loc"] == ["query", "q"]


def test_storage_failure_is_useful_503(tmp_path):
    database = tmp_path / "test.sqlite3"
    with TestClient(create_app(database, FakeEmbeddings())) as api:
        database.unlink()
        response = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"})
        assert response.status_code == 503
        assert response.json() == {"detail": "Storage temporarily unavailable"}


def test_search_persists_across_application_restart(tmp_path):
    database = tmp_path / "test.sqlite3"
    with TestClient(create_app(database, FakeEmbeddings())) as api:
        client = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@neviswealth.com"}).json()
        document = api.post(f"/clients/{client['id']}/documents", json={"title": "Bill", "content": "Water usage"}).json()
    with TestClient(create_app(database, FakeEmbeddings())) as api:
        assert api.get("/search", params={"q": "NevisWealth"}).json() == [{**client, "type": "client", "match_type": "literal"}]
        assert api.get("/search", params={"q": "Water usage"}).json() == [{**document, "type": "document", "match_type": "literal"}]


def test_document_chunk_limit_is_422_without_partial_write(tmp_path):
    class LimitedEmbeddings(FakeEmbeddings):
        def document(self, content):
            raise ValueError("Document exceeds the limit of 512 semantic chunks")

    with TestClient(create_app(tmp_path / "test.sqlite3", LimitedEmbeddings())) as api:
        client = api.post("/clients", json={"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"}).json()
        payload = {"title": "Too many chunks", "content": "Never saved"}
        response = api.post(f"/clients/{client['id']}/documents", json=payload)
        assert response.status_code == 422
        assert response.json()["detail"] == [{
            "type": "value_error", "loc": ["body", "content"],
            "msg": "Document exceeds the limit of 512 semantic chunks", "input": payload["content"],
        }]
        assert api.get("/search", params={"q": "Never saved"}).json() == []


def test_relevance_failure_is_503_without_internal_details(tmp_path):
    class FailingRelevance(FakeEmbeddings):
        def query(self, text):
            return [1.0, 0.0]

        def rerank(self, query, contents):
            raise RuntimeError('private model detail')

    with TestClient(create_app(tmp_path / 'unavailable.sqlite3', FailingRelevance())) as api:
        owner = api.post('/clients', json={
            'first_name': 'Ada', 'last_name': 'Lovelace', 'email': 'ada@example.com',
        }).json()
        assert api.post(f"/clients/{owner['id']}/documents", json={
            'title': 'Record', 'content': 'utility bill',
        }).status_code == 201
        response = api.get('/search', params={'q': 'address proof'})
        assert response.status_code == 503
        assert response.json() == {'detail': 'Semantic search temporarily unavailable'}


@pytest.mark.parametrize("email", ["JOHN@NEVISWEALTH.COM", " john@neviswealth.com "])
def test_duplicate_email_ignores_case_and_outer_spaces(api, email):
    payload = {"first_name": "John", "last_name": "Doe", "email": "John@NevisWealth.com"}
    created = api.post("/clients", json=payload)
    assert created.status_code == 201
    assert created.json()["email"] == "john@neviswealth.com"
    response = api.post("/clients", json={**payload, "first_name": "Jane", "email": email})
    assert response.status_code == 409
    assert response.json() == {"detail": "A client with this email already exists"}
    matches = api.get("/search", params={"q": "NevisWealth"}).json()
    assert len(matches) == 1
    assert matches[0]["id"] == created.json()["id"]
