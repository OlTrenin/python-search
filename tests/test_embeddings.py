import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from nevis.embeddings import Embeddings
from nevis.reranker import Reranker


class EmbeddingStartupTests(unittest.TestCase):
    def test_missing_model_fails_without_downloading(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'model'):
                Embeddings(Path(directory))

    def test_missing_relevance_model_fails_without_downloading(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'Missing relevance models'):
                Reranker(Path(directory))
