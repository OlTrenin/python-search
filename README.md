# Nevis Search API

I built a single search API for financial advisors to find clients by partial identifying details and documents by related concepts. `NevisWealth` finds `john.doe@neviswealth.com`; `address proof` finds a document containing `utility bill`.

Advisors can search in familiar terms without knowing the document's exact wording. I kept the system small: one Python service, one database, and local request processing.

## Run

Requirements: Docker with Compose v2, free port 8000, and internet access for the first build. No external API keys, GPU, or database server are needed.

```bash
docker compose up --build --wait
```

- API: http://localhost:8000
- Interactive Swagger: http://localhost:8000/docs
- OpenAPI 3.1 schema: http://localhost:8000/openapi.json
- Startup check: http://localhost:8000/health

I bundled both models in the image and pinned dependencies, model snapshots, and the base image digest. The first build downloads weights; requests run locally. The relevance model adds about 1.74 GB of weights. The test container used about 3.3 GiB of memory on the tested ARM64 machine. Allocate at least 4 GiB to Docker for one process, or about 8 GiB to run the API and real-model tests together. With less memory, stop the API during those tests.

Records and vectors live in the `nevis-data` named volume and survive restarts. `docker compose down` preserves the volume. **`docker compose down -v` deletes the data.**

The port binds to localhost. There is no authentication or advisor access isolation, so I use synthetic data for demonstrations.

## Try it

Create a client:

```bash
curl -sS http://localhost:8000/clients \
  -H 'Content-Type: application/json' \
  -d '{"first_name":"John","last_name":"Doe","email":"john.doe@neviswealth.com","description":"Advisor based in London","social_links":["@john"]}'
```

Replace `CLIENT_ID` with the returned `id`, then add a document and search:

```bash
CLIENT_ID=76e34999-35e4-4b92-bf53-ab7c8334aa98
curl -sS "http://localhost:8000/clients/$CLIENT_ID/documents" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Residence record","content":"utility bill"}'
curl -sSG http://localhost:8000/search --data-urlencode 'q=NevisWealth'
curl -sSG http://localhost:8000/search --data-urlencode 'q=address proof'
```

Example `NevisWealth` response captured during an HTTP check; new records receive different ids:

```json
[
  {
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@neviswealth.com",
    "description": "Advisor based in London",
    "social_links": ["@john"],
    "id": "76e34999-35e4-4b92-bf53-ab7c8334aa98",
    "type": "client",
    "match_type": "literal"
  }
]
```

Example `address proof` response:

```json
[
  {
    "title": "Residence record",
    "content": "utility bill",
    "id": "36b58948-5345-43c7-8c10-ff4138d84465",
    "client_id": "76e34999-35e4-4b92-bf53-ab7c8334aa98",
    "created_at": "2026-10-29T12:18:19.237871Z",
    "type": "document",
    "match_type": "semantic"
  }
]
```

The test corpus also includes `identity verification` for a passport photograph page and `employment income` for a payslip. `quantum entanglement` returns `[]` in this corpus.

## API contract

| Method and path | Input | Success | Errors |
|---|---|---|---|
| `POST /clients` | JSON: required `first_name`, `last_name`, `email`; optional `description`, `social_links` | `201`, client object | `409` duplicate email; `422` invalid input; `503` storage unavailable |
| `POST /clients/{id}/documents` | String client id; JSON with required `title`, `content` | `201`, document object | `404` client missing; `422` invalid input; `503` indexing or storage failure |
| `GET /search?q=...` | Required string `q` | `200`, array of clients and documents; `[]` for no match | `422` invalid query; `503` search or storage unavailable |
| `GET /health` | None | `200`, `{"status":"ok"}` after startup | Missing models or an incompatible index prevent startup |

A client contains `id`, `first_name`, `last_name`, `email`, `description`, and `social_links`. I use string UUIDs and default optional fields to `null` and `[]`. Social links are strings; URL format is not required.

A document contains `id`, `client_id`, `title`, `content`, and `created_at`. Creation time is timezone-aware UTC, and a foreign key protects the client relationship. Uploads accept text in JSON. File extraction and scan recognition are outside the API.

I added `type` (`client` or `document`) and `match_type` (`literal` or `semantic`) to search results so a consuming interface can distinguish entities and explain matches. Clients only use `literal`. Results include all entity fields; vectors and numeric scores are not exposed.

### Input rules

I trim and case-fold email before storing it. A unique database index prevents duplicate cards, including concurrent requests. One email belongs to one client.

Names, titles, and queries are trimmed; document content is preserved. Blank required strings, invalid email, unknown fields, and incorrect types return `422`.

| Field | Limit |
|---|---|
| First and last name | 200 characters each |
| Client description | 10,000 characters |
| Social links | 20 strings of 2,048 characters each |
| Document title | 500 characters |
| Document content | 100,000 characters and at most 512 chunks with either tokenizer |
| Search query | 500 characters, up to 510 embedding tokens; formatted relevance query up to 250 tokens |

These limits bound request memory and computation. Oversized input returns `422`; documents are never silently truncated.

Validation errors use FastAPI's `{"detail":[...]}` format. Unknown clients return `{"detail":"Client not found"}`; duplicate email returns `{"detail":"A client with this email already exists"}`. Internal failures return a generic `detail` and `503`, with diagnostics logged server-side. Literal search also depends on successful model processing; there is no fallback mode.

## Search behavior

For clients, I split the query on whitespace. Every part must occur in the same client's first name, last name, email, or description. Unicode `casefold` makes matching case-insensitive; part order does not matter. `John NevisWealth` finds John Doe at the matching domain and excludes Jane at that domain. Social links and attached documents do not supply client matches.

For documents, literal search uses the entire query in the title or content. `%`, `_`, and apostrophes are literal characters. There is no query-operator syntax.

Semantic search uses document content:

1. `BAAI/bge-small-en-v1.5` through FastEmbed produces normalized vectors. A document becomes a candidate when its best-chunk cosine is at least `0.4`.
2. `mixedbread-ai/mxbai-rerank-large-v1` scores the query together with candidate passages. Every query uses the same context: `Documents related to a client's {q}.`.
3. A document is included when its best passage score is at least `-6.0`. This is a raw logit, not a probability.

Both tokenizers split content at line boundaries, then into windows of up to 256 tokens with 32-token overlap. Identical passages are scored once. Titles participate only in literal search, so a generic title does not supply semantic relevance.

Results contain literal clients first, literal documents next, then semantic documents by descending relevance score. Literal groups use id order; ids break semantic ties. Each entity appears once. There is no pagination or result-count limit.

I define relevance as a connection to the topic. `address proof` includes both `utility bill` and `The client has not provided a utility bill yet.`, while excluding `We need to prove that an algorithm is correct.`. Retrieval does not certify that address evidence has been supplied or is valid. The search code contains no special cases for these strings.

## Architecture choices

I chose one service for three API operations and a small corpus. Separate services would add deployment and maintenance work to this scope.

```text
HTTP -> FastAPI/Pydantic -> SQLite
              |
              +-> local models -> document vectors

Search -> literal matches + vector candidates -> relevance scoring -> result array
```

The [design](docs/design.md) explains the decisions in detail.

| Choice | Why I made it | Alternative and trade-off |
|---|---|---|
| Python 3.12, FastAPI, Pydantic | Input/output validation, typed responses, and Swagger in one place | Flask would need these pieces assembled separately |
| SQLite through standard `sqlite3` | Records, documents, and vectors in one database without a database server | PostgreSQL handles greater write concurrency but adds an operational component |
| Synchronous indexing before storage | `201` means the document and vectors are stored together and searchable | A queue could shorten upload waits but needs indexing states, retries, and lag handling |
| Literal and semantic search | Exact identifiers remain searchable alongside related terms | A synonym dictionary needs manual maintenance; embeddings alone can confuse word senses |
| Relevance scoring | Separates address evidence from algorithm proofs in the tested cases | Cosine alone failed this distinction; tested smaller and quantized models did not retain the required separation |
| Local ONNX/FastEmbed models | No credentials or network call per search; text stays local during processing | An external service reduces image size but adds data transfer, cost, and network dependence |
| Exact vector scan | Avoids a vector database for the current corpus | Cost grows linearly with the total chunk count; an index addresses candidate-selection latency |
| Separate `runtime` and `test` targets | Reproducible tests without test tools in the runtime image | Two build targets share the same dependencies and models |

One transaction stores each document and its vectors. The database records the embedding and chunking version; incompatible versions stop startup. Changing embeddings or chunking requires explicit reindexing. There is no reindexing tool yet.

On an older database, startup normalizes email and creates the unique index transactionally. Conflicts stop startup without deleting or merging records. Back up the database before resolving duplicates manually.

## Verify

Build the test image and run the suite offline with bundled models:

```bash
docker build --target test -t nevis-search-api:test .
docker run --rm --network none nevis-search-api:test pytest -q
```

For API, storage, and validation checks without model inference:

```bash
docker run --rm --network none nevis-search-api:test pytest -q -m 'not semantic'
```

The `semantic` tests use real models. Other tests use substitute vectors to isolate API behavior, storage, and ordering. The full model suite can take several minutes on CPU.

Reproduce per-pair relevance scores and the previous cosine-only baseline:

```bash
docker run --rm --network none nevis-search-api:test python -m scripts.evaluate_relevance
```

Check the running HTTP service and verify that ids survive a restart:

```bash
docker compose exec -T api python - < scripts/smoke.py
docker compose restart api
docker compose up --wait
docker compose exec -T api python - --verify-restart < scripts/smoke.py
```

The smoke script creates synthetic records and stores ids in `/data/smoke.json`. Repeated runs reuse the demo client, check duplicate-email rejection, and add documents. Run it against a demonstration database.

### Relevance evaluation

The evaluation contains 41 synthetic pairs: 15 calibration, 15 validation, and 11 final holdout pairs. Parameters were frozen before validation and final holdout. The current pipeline had no false positives or false negatives on these sets; the previous pipeline produced `3/1`, `3/0`, and `3/1`, respectively. Cases are in [tests/relevance_cases.json](tests/relevance_cases.json).

The models target English text. A short topic inside a long mixed-topic paragraph can be diluted during relevance scoring.

## Scope

I deliver source code, Docker Compose, core and edge-case tests, API examples, and Swagger. The API creates clients and text documents and searches both. Unique email, AND matching across client query parts, and disambiguation of `proof` are product decisions added to the base contract.

Document summaries and external deployment are outside this version. Editing, deletion, file storage, queues, and caching are also omitted.
