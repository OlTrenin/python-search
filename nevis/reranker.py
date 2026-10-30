"""Pinned local cross-encoder; scores are logits, not calibrated probabilities."""

import math
import os
from pathlib import Path
from threading import Lock

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from tokenizers import Tokenizer

from nevis.chunks import chunks

RERANKER_PATH = Path(os.environ.get('RERANKER_PATH', 'models/rerankers'))
MODEL_NAME = 'mixedbread-ai/mxbai-rerank-large-v1'
MODEL_REVISION = '98f655841d5caf0b16eaff79c2b4ca109d920d17'
# shortcut: cutoffs are selected on a small labeled corpus; recalibrate before real advisor use.
RERANK_THRESHOLD = -6.0


class Reranker:
    def __init__(self, path=RERANKER_PATH):
        required = ['tokenizer.json', 'onnx/model.onnx']
        if not all((path / name).is_file() for name in required):
            raise RuntimeError(f'Missing relevance models in {path}; run python -m nevis.reranker')
        self.tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        TextCrossEncoder.add_custom_model(
            model=MODEL_NAME, sources=ModelSource(hf=MODEL_NAME), model_file='onnx/model.onnx',
        )
        self.model = TextCrossEncoder(model_name=MODEL_NAME, specific_model_path=str(path),
                                     local_files_only=True, threads=2)
        self.lock = Lock()

    def validate_query(self, query):
        # Reserve 256 passage tokens and the pair separator; never let FastEmbed truncate a pair.
        if len(self.tokenizer.encode(f"Documents related to a client's {query}.").ids) > 250:
            raise ValueError('Query exceeds the relevance model token limit')

    def rerank(self, query, contents):
        for content in contents:
            with self.lock:
                passages = list(dict.fromkeys(chunks(content, self.tokenizer)))
                query_text = f"Documents related to a client's {query}."
                if any(len(self.tokenizer.encode(query_text, passage).ids) > 512 for passage in passages):
                    raise ValueError('Relevance model context limit exceeded')
                scores = list(self.model.rerank(query_text, passages, batch_size=1))
                if len(scores) != len(passages) or any(not math.isfinite(score) for score in scores):
                    raise RuntimeError('Invalid relevance model scores')
            yield max(scores, default=-math.inf)


if __name__ == '__main__':
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_NAME, revision=MODEL_REVISION, local_dir=RERANKER_PATH,
                      allow_patterns=['*.json', '*.txt', 'onnx/model.onnx'])
    Reranker()
