"""Pinned, local-only English embeddings; model download is an explicit build step."""

import os
from pathlib import Path

from nevis.chunks import chunks

MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MODEL_REPO = 'Qdrant/bge-small-en-v1.5-onnx-Q'
MODEL_REVISION = 'aa8f8b060edb00e03bfdd08813a2949946c8ba55'
MODEL_PATH = Path(os.environ.get('MODEL_PATH', 'models/bge-small-en-v1.5'))
INDEX_VERSION = f'{MODEL_REVISION}:lines-chunks256-overlap32-content:v1'
# shortcut: cutoff calibrated on the homework corpus, reassess on labeled advisor queries before production.
SIMILARITY_THRESHOLD = 0.4


class Embeddings:
    def __init__(self, path: Path = MODEL_PATH):
        if not all((path / name).is_file() for name in ('tokenizer.json', 'model_optimized.onnx')):
            raise RuntimeError(f'Missing semantic model in {path}; run python -m nevis.embeddings')
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer
        from threading import Lock
        from nevis.reranker import Reranker

        self.reranker = Reranker()
        self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.model = TextEmbedding(
            model_name=MODEL_NAME, specific_model_path=str(path),
            local_files_only=True, threads=2,
        )
        self.lock = Lock()

    def query(self, text: str) -> list[float]:
        if len(self.tokenizer.encode(text, add_special_tokens=False).ids) > 510:
            raise ValueError('Query exceeds the semantic model token limit')
        self.reranker.validate_query(text)
        with self.lock:
            return next(self.model.embed([text])).tolist()

    def document(self, content: str) -> list[list[float]]:
        passages = chunks(content, self.tokenizer)
        chunks(content, self.reranker.tokenizer)
        with self.lock:
            return [vector.tolist() for vector in self.model.embed(passages, batch_size=16)]

    def rerank(self, query, contents):
        return self.reranker.rerank(query, contents)


if __name__ == '__main__':
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO, revision=MODEL_REVISION, local_dir=MODEL_PATH,
        allow_patterns=['*.json', 'model_optimized.onnx', 'README.md'],
    )
