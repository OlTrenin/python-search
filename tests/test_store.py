import tempfile
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import unittest
from pathlib import Path
from uuid import UUID

from nevis.store import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "records.sqlite3"
        self.store = Store(self.path, "test-model-v1")

    def test_client_defaults_unique_email_and_persistence(self):
        data = {"first_name": "John", "last_name": "Doe", "email": "john@neviswealth.com"}
        client = self.store.create_client(data)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.create_client(data)
        UUID(client["id"])
        self.assertEqual(client, {**data, "id": client["id"], "description": None, "social_links": []})
        reopened = Store(self.path, "test-model-v1")
        self.assertTrue(reopened.client_exists(client["id"]))
        self.assertFalse(reopened.client_exists("absent"))

    def test_client_substring_fields_unicode_and_literal_punctuation(self):
        client = self.store.create_client({
            "first_name": "Jörg", "last_name": "Straße", "email": "john.doe@neviswealth.com",
            "description": "Advisor for 50%_holdings and O'Reilly", "social_links": ["@joerg"],
        })
        for query in ["JÖ", "STRASSE", "jÖrg sTRASSE", "NevisWealth", "VISOR", "%_", "O'Reilly"]:
            with self.subTest(query=query):
                results = self.store.search(query, [1.0, 0.0], 0.55)
                self.assertEqual(results, [{**client, "type": "client", "match_type": "literal"}])
        self.assertEqual(self.store.search("' OR 1=1 --", [1.0, 0.0], 0.55), [])
        self.assertEqual(self.store.search("quantum entanglement", [1.0, 0.0], 0.55), [])

    def test_document_creation_timestamp_association_and_missing_client(self):
        client = self.store.create_client({"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"})
        data = {"title": "Utility bill", "content": "Water usage for September"}
        before = datetime.now(timezone.utc)
        document = self.store.create_document(client["id"], data, [[1.0, 0.0]])
        UUID(document["id"])
        created = datetime.fromisoformat(document["created_at"])
        self.assertLessEqual(before, created)
        self.assertLessEqual(created, datetime.now(timezone.utc))
        self.assertIsNotNone(created.tzinfo)
        self.assertEqual(document["client_id"], client["id"])
        self.assertEqual(document["title"], data["title"])
        self.assertEqual(document["content"], data["content"])
        self.assertNotIn("vectors", document)
        with self.assertRaises(LookupError):
            self.store.create_document("missing-client", {"title": "Orphan", "content": "Not saved"}, [[1.0, 0.0]])

    def test_document_literal_semantic_ranking_deduplication_and_restart(self):
        client = self.store.create_client({"first_name": "Shared", "last_name": "Client", "email": "a@example.com"})
        other_client = self.store.create_client({"first_name": "Shared", "last_name": "Other", "email": "b@example.com"})
        title_hit = self.store.create_document(client["id"], {"title": "SHARED title", "content": "Other text"}, [[1.0, 0.0]])
        content_hit = self.store.create_document(client["id"], {"title": "Other title", "content": "shared content"}, [[0.0, 1.0]])
        high = self.store.create_document(client["id"], {"title": "High similarity", "content": "Related concept"}, [[0.0, 1.0], [0.9, math.sqrt(0.19)]])
        tie_a = self.store.create_document(client["id"], {"title": "Related A", "content": "Concept A"}, [[0.8, 0.6]])
        tie_b = self.store.create_document(client["id"], {"title": "Related B", "content": "Concept B"}, [[0.8, 0.6]])
        rejected = self.store.create_document(client["id"], {"title": "Unrelated", "content": "Different topic"}, [[0.5, math.sqrt(0.75)]])
        results = self.store.search("shared", [1.0, 0.0], 0.55)
        expected_ids = (sorted([client["id"], other_client["id"]])
                        + sorted([title_hit["id"], content_hit["id"]])
                        + [high["id"]] + sorted([tie_a["id"], tie_b["id"]]))
        self.assertEqual([result["id"] for result in results], expected_ids)
        self.assertEqual([result["type"] for result in results], ["client"] * 2 + ["document"] * 5)
        self.assertEqual([result["match_type"] for result in results], ["literal"] * 4 + ["semantic"] * 3)
        self.assertEqual(len({result["id"] for result in results}), len(results))
        self.assertNotIn(rejected["id"], expected_ids)
        for result in results:
            self.assertNotIn("vectors", result)
            self.assertNotIn("score", result)
        reopened = Store(self.path, "test-model-v1")
        self.assertEqual(reopened.search("shared", [1.0, 0.0], 0.55), results)
        self.assertEqual(reopened.search("missing", [-1.0, 0.0], 0.55), [])

    def test_index_version_mismatch_fails_without_changing_saved_data(self):
        client = self.store.create_client({"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"})
        with self.assertRaisesRegex(RuntimeError, "reindex"):
            Store(self.path, "different-model")
        self.assertEqual(Store(self.path, "test-model-v1").search("Ada", [1.0, 0.0], 0.55)[0]["id"], client["id"])

    def test_failed_document_write_does_not_leave_partial_document(self):
        client = self.store.create_client({"first_name": "Ada", "last_name": "Lovelace", "email": "ada@example.com"})
        with self.assertRaises(TypeError):
            self.store.create_document(client["id"], {"title": "Broken indexing", "content": "Never persisted"}, [{1.0}])
        self.assertEqual(self.store.search("Never persisted", [-1.0, 0.0], 0.55), [])
        self.assertFalse(self.store.client_exists("' OR 1=1 --"))
        with self.assertRaises(LookupError):
            self.store.create_document("' OR 1=1 --", {"title": "Orphan", "content": "Not saved"}, [[1.0, 0.0]])
        self.assertEqual(self.store.search("Not saved", [-1.0, 0.0], 0.55), [])

    def test_unique_email_is_case_insensitive_and_race_safe(self):
        data = {"first_name": "John", "last_name": "Doe", "email": " John@NevisWealth.com "}

        def create():
            try:
                return self.store.create_client(data)
            except sqlite3.IntegrityError:
                return None

        with ThreadPoolExecutor(max_workers=2) as workers:
            results = list(workers.map(lambda _: create(), range(2)))
        self.assertEqual(sum(result is not None for result in results), 1)
        matches = Store(self.path, "test-model-v1").search("NevisWealth", [-1.0, 0.0], 0.58)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["email"], "john@neviswealth.com")

    def test_legacy_database_is_normalized_without_losing_documents(self):
        data = {"first_name": "John", "last_name": "Doe", "email": "john@neviswealth.com"}
        client = self.store.create_client(data)
        document = self.store.create_document(client["id"], {"title": "Bill", "content": "utility bill"}, [[1.0, 0.0]])
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP INDEX clients_email_unique")
            connection.execute("UPDATE clients SET email = ? WHERE id = ?", ("John@NevisWealth.com", client["id"]))
        reopened = Store(self.path, "test-model-v1")
        self.assertEqual(reopened.search("john@neviswealth.com", [-1.0, 0.0], 0.58)[0]["email"], data["email"])
        self.assertEqual(reopened.search("utility bill", [-1.0, 0.0], 0.58)[0]["id"], document["id"])
        with self.assertRaises(sqlite3.IntegrityError):
            reopened.create_client({**data, "email": "JOHN@NEVISWEALTH.COM"})

    def test_legacy_duplicate_emails_stop_startup_without_data_loss(self):
        data = {"first_name": "John", "last_name": "Doe", "email": "john@neviswealth.com"}
        client = self.store.create_client(data)
        document = self.store.create_document(client["id"], {"title": "Bill", "content": "utility bill"}, [[1.0, 0.0]])
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP INDEX clients_email_unique")
            connection.execute(
                "INSERT INTO clients SELECT ?, first_name, last_name, ?, description, social_links FROM clients WHERE id = ?",
                ("legacy-duplicate", "John@NevisWealth.com", client["id"]),
            )
            clients_before = connection.execute("SELECT * FROM clients ORDER BY id").fetchall()
        with self.assertRaisesRegex(RuntimeError, "duplicate emails"):
            Store(self.path, "test-model-v1")
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT * FROM clients ORDER BY id").fetchall(), clients_before)
            self.assertEqual(connection.execute("SELECT id FROM documents").fetchall(), [(document["id"],)])


if __name__ == "__main__":
    unittest.main()
