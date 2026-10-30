"""Reproduce labeled retrieval scores: python -m scripts.evaluate_relevance."""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from nevis.embeddings import Embeddings, MODEL_REVISION as EMBEDDING_REVISION, SIMILARITY_THRESHOLD
from nevis.reranker import MODEL_NAME, MODEL_REVISION, RERANK_THRESHOLD


def main():
    data = (Path(__file__).resolve().parents[1] / 'tests/relevance_cases.json').read_bytes()
    model = Embeddings()
    result = {'model': MODEL_NAME, 'revision': MODEL_REVISION, 'embedding_revision': EMBEDDING_REVISION,
              'dataset_sha256': hashlib.sha256(data).hexdigest(),
              'candidate_threshold': SIMILARITY_THRESHOLD, 'rerank_threshold': RERANK_THRESHOLD,
              'query_template': "Documents related to a client's {}.", 'batch_size': 1, 'splits': {}}
    failures = 0
    for split, cases in json.loads(data).items():
        rows = []
        for case in cases:
            query = case['q']
            query_vector = model.query(query)
            for relevant, contents in [(True, case['relevant']), (False, case['irrelevant'])]:
                for content in contents:
                    assert query.casefold() not in content.casefold(), 'Evaluation must exercise semantic matches'
                    vectors = model.document(content)
                    cosine = max(math.fsum(a * b for a, b in zip(query_vector, vector, strict=True))
                                 for vector in vectors)
                    started = perf_counter()
                    score = list(model.rerank(query, [content]))[0]
                    selected = cosine >= SIMILARITY_THRESHOLD and score >= RERANK_THRESHOLD
                    rows.append({'q': query, 'content': content, 'relevant': relevant, 'cosine': cosine,
                                 'rerank_score': score, 'selected': selected, 'baseline_selected': cosine >= 0.58,
                                 'rerank_seconds': perf_counter() - started})
        metrics = {}
        for label, field in [('baseline', 'baseline_selected'), ('current', 'selected')]:
            counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            for row in rows:
                counts[('t' if row[field] == row['relevant'] else 'f') + ('p' if row[field] else 'n')] += 1
            metrics[label] = counts
        failures += metrics['current']['fp'] + metrics['current']['fn']
        result['splits'][split] = {'metrics': metrics, 'rows': rows}
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
