import json
import math
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


class Store:
    def __init__(self, path: str | Path, index_version: str):
        self.path = path
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS clients (
                    id TEXT PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    description TEXT,
                    social_links TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL REFERENCES clients(id),
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    vectors TEXT NOT NULL
                );
            """)
            connection.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('index_version', ?)", (index_version,)
            )
            stored_version = connection.execute(
                "SELECT value FROM metadata WHERE key = 'index_version'"
            ).fetchone()[0]
            if stored_version != index_version:
                raise RuntimeError("Stored search index version differs; reindex before changing the model")

            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = 'clients_email_unique'"
            ).fetchone() is None:
                emails = [(row["email"].strip().casefold(), row["id"])
                          for row in connection.execute("SELECT id, email FROM clients")]
                try:
                    connection.executemany("UPDATE clients SET email = ? WHERE id = ?", emails)
                    connection.execute("CREATE UNIQUE INDEX clients_email_unique ON clients(email)")
                except sqlite3.IntegrityError:
                    raise RuntimeError(
                        "Existing clients have duplicate emails; resolve duplicates before starting. "
                        "No client or document data was deleted."
                    ) from None

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def create_client(self, data: dict) -> dict:
        client = {"description": None, "social_links": [], **data, "id": str(uuid4()),
                  "email": data["email"].strip().casefold()}
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?)",
                (client["id"], client["first_name"], client["last_name"], client["email"],
                 client["description"], json.dumps(client["social_links"])),
            )
        return client

    def client_exists(self, client_id: str) -> bool:
        with self._connection() as connection:
            return connection.execute("SELECT 1 FROM clients WHERE id = ?", (client_id,)).fetchone() is not None

    def create_document(self, client_id: str, data: dict, vectors: list[list[float]]) -> dict:
        document = {**data, "id": str(uuid4()), "client_id": client_id,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        with self._connection() as connection:
            if connection.execute("SELECT 1 FROM clients WHERE id = ?", (client_id,)).fetchone() is None:
                raise LookupError("Client not found")
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (document["id"], client_id, document["title"], document["content"],
                 document["created_at"], json.dumps(vectors)),
            )
        return document

    def search(self, q: str, query_vector: list[float], threshold: float) -> list[dict]:
        query = q.casefold()
        parts = query.split()
        with self._connection() as connection:
            clients = connection.execute("SELECT * FROM clients ORDER BY id").fetchall()
            documents = connection.execute("SELECT * FROM documents ORDER BY id").fetchall()
        results = []
        for row in clients:
            client = dict(row)
            fields = [field.casefold() for field in (
                client["first_name"], client["last_name"], client["email"], client["description"] or "",
            )]
            if all(any(part in field for field in fields) for part in parts):
                client["social_links"] = json.loads(client["social_links"])
                results.append({**client, "type": "client", "match_type": "literal"})
        semantic_matches = []
        # shortcut: exact scan suits the homework corpus; use an index when measured latency exceeds the budget.
        for row in documents:
            document = dict(row)
            vectors = json.loads(document.pop("vectors"))
            if query in document["title"].casefold() or query in document["content"].casefold():
                results.append({**document, "type": "document", "match_type": "literal"})
            else:
                score = max((math.fsum(a * b for a, b in zip(query_vector, vector, strict=True))
                             for vector in vectors), default=-1.0)
                if score >= threshold:
                    semantic_matches.append((score, document))
        semantic_matches.sort(key=lambda match: (-match[0], match[1]["id"]))
        results.extend({**document, "type": "document", "match_type": "semantic"}
                       for _, document in semantic_matches)
        return results
