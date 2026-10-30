# Solution design

## Context and goal

I built search around two advisor actions: finding a client from part of their identifying details and finding a document by topic when its wording differs from the query. `NevisWealth` finds `john.doe@neviswealth.com`; `address proof` finds `utility bill`.

I chose one synchronous Python service, local models, and one database. The design uses one API process and an exact vector scan for a small document corpus.

## Scope

I implemented client creation, text-document creation, and unified search. Documents receive UTC creation timestamps, search returns an entity array, and records survive restarts.

Document summaries, a frontend, external deployment, and background indexing are outside this version. The service binds locally and uses synthetic demonstration data; authentication and advisor access isolation are absent.

## Decisions

| Choice | Why I made it | Alternative and trade-off |
|---|---|---|
| Python 3.12, FastAPI, Pydantic | Request/response validation, typed contracts, and Swagger in one place | Flask would require validation and API documentation to be assembled separately |
| SQLite through standard `sqlite3`, one connection per operation | Persistent storage and atomic writes without a database server | PostgreSQL supports greater write concurrency but adds an operational component. SQLite serializes writes |
| `BAAI/bge-small-en-v1.5` through FastEmbed/ONNX | Related-concept retrieval without credentials or per-request charges; no PyTorch or GPU | A synonym dictionary needs a maintained list. External embeddings reduce local computation but add text transfer and service availability requirements. The selected model targets English |
| Literal matching, BGE candidates, and `mxbai-rerank-large-v1` FP32 | Preserve exact identifiers and distinguish word senses through relevance scoring | Embeddings alone confused the tested meanings of `proof`. Tested smaller and quantized models did not preserve the required separation. The relevance model adds about 1.74 GB of weights and CPU cost |
| Exact cosine comparison of passage vectors | Avoid a separate vector index for the current corpus | Scan cost grows linearly with total passage count. An index addresses candidate-selection latency; reranker cost is a separate part of search |

### Combining literal and semantic search

Literal search gives predictable matches for names, emails, and exact phrases. Semantic search connects different terms for a topic. Query expansion through a synonym dictionary would be compact, but each relationship would need manual maintenance.

Cosine similarity alone retrieved an algorithm proof for an address-evidence query. I added query/passage relevance scoring to distinguish those contexts. Raising the cosine cutoff also removed relevant documents.

Local inference removes a network call from each search and keeps request text within the service. I accept a larger image and greater memory use in exchange. Pinned model versions make builds and checks reproducible.

## Data and document writes

```text
Create client -> validation -> SQLite

Add document -> validation -> local passage vectors
                           -> one transaction: document + vectors

Search -> literal matches + cosine candidates
       -> candidate relevance scores -> typed entity array
```

| Table | Fields |
|---|---|
| `clients` | String UUID `id`, `first_name`, `last_name`, `email`, nullable `description`, JSON `social_links` |
| `documents` | String UUID `id`, foreign key `client_id`, `title`, `content`, UTC `created_at`, JSON `vectors` |

I compute vectors before the short write transaction. If inference fails, no document is stored. A `201` response means the text and vectors were committed together and are searchable. A queue could reduce upload waits, but would require indexing states, retries, and lag handling.

Foreign keys are enabled, and connections wait when the database is busy. SQL queries are parameterized. Python `casefold` provides Unicode substring comparison; `%` and `_` do not become SQL wildcard operators.

### Email uniqueness

I use one client card per email. Before storage, I trim surrounding whitespace and apply `casefold`. A unique SQLite index enforces the rule during concurrent inserts; duplicates return `409`. A shared address cannot belong to several clients.

For older databases, normalization and index creation run in one transaction. A collision stops startup and rolls back changes without deleting or merging clients. Back up the database before resolving duplicate records manually.

## Search rules

### Client cards

I split the query on whitespace. Every part must occur in `first_name`, `last_name`, `email`, or `description` of the same client. Case and part order do not matter; punctuation is literal.

`John NevisWealth` finds John Doe at `john.doe@neviswealth.com` and excludes Jane at that domain. Full names match across first and last name fields. Parts cannot be combined across clients, social links, or attached documents. This lets advisors narrow a search using several known details.

### Documents and passages

Literal document search uses the full `q` in title or content. Models also receive the full query. Titles are excluded from semantic scoring so a generic title does not supply relevance for unrelated content.

Both tokenizers split content at line boundaries, then into windows of up to 256 tokens with 32-token overlap. Passage extraction uses original character offsets without silent truncation. More than 512 passages from either tokenizer returns `422` before embedding inference.

A best-passage cosine of at least `0.4` selects a candidate. The cross-encoder then scores the query with each distinct content passage; identical passages are scored once. A document's score is its maximum passage score, and inclusion requires at least `-6.0`.

I limit each model to two CPU threads. The reranker processes one passage per batch.

### Query context and calibration

Every reranker query uses `Documents related to a client's {q}.`. The search code contains no conditions for known query strings.

I selected the `-6.0` threshold on 15 calibration pairs: the lowest positive score was `-5.302`, and the highest negative score was `-6.712`. The selected parameters were then checked on 15 validation and 11 new final holdout pairs without changing the threshold.

An earlier holdout used to evaluate the unsuccessful Jina candidate became the development-validation set. Only the final 11-pair set remained the final holdout. The [evaluation script](../scripts/evaluate_relevance.py) reproduces current scores and the cosine-only baseline.

Cosine values and raw logits are not probabilities. The formatted query is limited to 250 reranker tokens, reserving context for a 256-token passage and separators. The 510-token BGE and 500-character query limits also apply. Exceeding a limit returns `422`.

### Meaning of relevance

I retrieve topical content. `address proof` includes both `utility bill` and `The client has not provided a utility bill yet.`, and excludes `We need to prove that an algorithm is correct.`.

Retrieval does not certify that address evidence has been supplied or is valid. That decision belongs to the advisor or a separate document-validation process.

## Responses, validation, and failures

I preserve the client and document fields in the base contract. Search adds `type` and `match_type` so an interface can distinguish entities and explain matches. Vectors and numeric scores remain internal.

Literal clients precede literal documents, both in id order. Semantic documents follow in descending reranker score order, with ids breaking ties. Each entity appears once; no matches returns `[]`.

Names, titles, and queries are trimmed while document content is preserved. Blank required strings, invalid email, unknown fields, incorrect types, and exceeded limits return `422`. Social links remain strings without a URL requirement. Defaults and field limits are in the [README](../README.md).

Unknown clients return `404`; duplicate email returns `409`. Model or storage failures during requests return a generic `503`, with diagnostics logged server-side. Literal results also depend on successful model processing.

## Packaging and verification

I split Docker builds into `runtime` and `test`. The runtime contains application code, runtime dependencies, and both models. The test target adds pytest, tests, labeled cases, and verification scripts. `requirements-test.txt` extends `requirements.txt`; test tools stay out of the default image.

The build downloads fixed Hugging Face commits for the Qdrant BGE model and `mxbai-rerank-large-v1`. Runtime loads local files only, and missing model files stop startup. The first build needs internet; execution was checked with networking disabled.

- SQLite tests and substitute vectors isolate persistence, ordering, and failure behavior.
- FastAPI TestClient checks HTTP contracts and validation.
- Real models check semantic scenarios against labeled cases.
- HTTP checks against Compose and restart checks exercise the service and persisted ids.

Labeled data is in [tests/relevance_cases.json](../tests/relevance_cases.json); reproduction commands are in the [README](../README.md). Generated reports, caches, and experimental models are excluded from delivery.

## Resources and long documents

The FP32 relevance model adds about 1.74 GB of weights. The test container used about 3.3 GiB of memory; the full real-model suite took about seven minutes on the available CPU. Tested quantized variants lost the required separation, and some scores varied with batch size. Changing precision requires recalibration and validation.

Long documents increase computation. Taking the maximum across passages can increase false matches, while a short topic within a mixed-topic paragraph can be diluted.

## Updates and persistence

I initialize the schema at startup and store data in a named volume. Restarting and `docker compose down` preserve records. **`docker compose down -v` deletes the data volume.**

SQLite records the embedding and chunking version. An incompatible version stops startup to prevent comparisons between different vector spaces. Changing embeddings or windows requires explicit reindexing; no reindexing tool is included. Changing only the reranker leaves stored vectors intact but requires score and threshold evaluation.

Email uniqueness upgrades run transactionally at startup. Existing duplicates require manual resolution after a backup. Other schema changes require an explicit migration before reusing stored data.
