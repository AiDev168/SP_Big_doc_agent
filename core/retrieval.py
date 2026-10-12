"""Multi-stage retrieval: multi-query, hybrid (vector + FTS5), reranking.
"""

import logging
from typing import List, Optional

from localrag.core.compat import CrossEncoder, Document
from localrag.core.config import Config, resolve_local_model
from localrag.core.store import PersistentStore
from localrag.core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """Configurable, multi-stage retrieval pipeline built on top of the
    vector store:

      1. (optional) Multi-query expansion — asks the LLM for alternate
         phrasings of the question so retrieval isn't at the mercy of a
         single wording.
      2. Hybrid fetch — for each query variant, blend dense vector search
         with sparse BM25 keyword search (EnsembleRetriever), so exact
         terms/names/codes that embeddings can miss still surface.
      3. Merge + dedupe candidates across all query variants.
      4. (optional) Cross-encoder reranking — a small model scores every
         candidate directly against the *original* question and keeps only
         the top few, which is far more precise than raw vector distance.

    Every stage degrades gracefully: if a dependency isn't installed or a
    stage is toggled off, the pipeline just falls through to plain vector
    search on the original query.

    All stages accept an optional file_names list that restricts retrieval
    to chunks from those documents only (None = all documents).
    """

    def __init__(self, config: Config, vector_manager: "VectorStoreManager", chunk_store: PersistentStore):
        self.config = config
        self.vector_manager = vector_manager
        self.chunk_store = chunk_store
        self._cross_encoder = None
        self._cross_encoder_name = None

    def mark_dirty(self) -> None:
        # No-op with the SQLite/FTS5-backed store: search_keyword() always
        # reads the current on-disk index directly, so there is nothing to
        # rebuild or invalidate. Kept as a method so callers written against
        # the old in-memory-index API don't need to change.
        pass

    def _get_cross_encoder(self):
        if CrossEncoder is None:
            return None
        model_name = self.config.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        if self._cross_encoder is None or self._cross_encoder_name != model_name:
            try:
                self._cross_encoder = CrossEncoder(
                    resolve_local_model(model_name, self.config.models_dir, "rerankers")
                )
                self._cross_encoder_name = model_name
            except Exception:
                logger.exception(
                    f"Failed to load cross-encoder '{model_name}' — reranking is skipped. "
                    f"Offline: put the model folder in {self.config.models_dir / 'rerankers'}."
                )
                self._cross_encoder = None
        return self._cross_encoder

    @staticmethod
    def _doc_key(doc) -> tuple:
        return (doc.metadata.get("file_name", ""), hash(doc.page_content))

    def _fetch_candidates(self, query: str, k: int, file_names: Optional[List[str]] = None) -> List:
        vector_retriever = self.vector_manager.get_retriever(k=k, file_names=file_names)
        vector_docs: List = []
        if vector_retriever is not None:
            try:
                vector_docs = vector_retriever.invoke(query)
            except Exception:
                logger.exception("Vector retrieval failed")
                vector_docs = []

        use_hybrid = self.config.config.get("use_hybrid_search", True)
        keyword_docs: List = (
            self.chunk_store.search_keyword(query, k=k, file_names=file_names) if use_hybrid else []
        )

        if not keyword_docs:
            return vector_docs
        if not vector_docs:
            return keyword_docs

        # Reciprocal rank fusion: each list contributes 1/(rank_constant +
        # rank) to a shared score, weighted by hybrid_vector_weight. This
        # combines the two ranked lists without needing comparable raw
        # scores (vector distance and BM25 score are on different scales),
        # and — unlike the old EnsembleRetriever — needs nothing held in
        # memory beyond these two already-bounded (size k) result lists.
        weight = min(max(float(self.config.config.get("hybrid_vector_weight", 0.6)), 0.0), 1.0)
        rank_constant = 60
        scores: dict = {}
        doc_by_key: dict = {}
        for rank, doc in enumerate(vector_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rank_constant + rank + 1))
        for rank, doc in enumerate(keyword_docs):
            key = self._doc_key(doc)
            doc_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + (1.0 - weight) * (1.0 / (rank_constant + rank + 1))

        ordered_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [doc_by_key[key] for key in ordered_keys[:k]]

    def _expand_queries(self, query: str, llm) -> List[str]:
        if not self.config.config.get("use_multi_query", False) or llm is None:
            return [query]

        n = int(self.config.config.get("multi_query_count", 3))
        expansion_prompt = (
            f"Generate {n} different ways to phrase the following question, each "
            f"capturing the same intent but with different wording or emphasis. "
            f"Return exactly {n} lines, one question per line, with no numbering, "
            f"bullets, or extra commentary. Respond in the same language as the "
            f"question.\n\nQuestion: {query}"
        )
        try:
            raw = llm.invoke(expansion_prompt)
            text = raw if isinstance(raw, str) else str(raw)
            variants = [line.strip(" \t-•*") for line in text.splitlines() if line.strip()]
            variants = [v for v in variants if v and v.lower() != query.lower()][:n]
            return [query] + variants
        except Exception:
            logger.exception("Multi-query expansion failed, falling back to the original query only")
            return [query]

    def _rerank(self, query: str, documents: List, top_k: int) -> List:
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is None or not documents:
            return documents[:top_k]
        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = cross_encoder.predict(pairs)
            ranked = [doc for _, doc in sorted(zip(scores, documents), key=lambda pair: pair[0], reverse=True)]
            return ranked[:top_k]
        except Exception:
            logger.exception("Cross-encoder reranking failed, returning unranked candidates")
            return documents[:top_k]

    def retrieve(self, query: str, llm=None, file_names: Optional[List[str]] = None) -> List:
        candidate_k = int(self.config.config.get("rerank_candidates", 20))
        final_k = int(self.config.config.get("final_k", 5))

        queries = self._expand_queries(query, llm)

        seen = set()
        merged: List = []
        for q in queries:
            for doc in self._fetch_candidates(q, k=candidate_k, file_names=file_names):
                key = (doc.metadata.get("file_name", ""), hash(doc.page_content))
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)

        if self.config.config.get("use_reranker", True):
            return self._rerank(query, merged, final_k)
        return merged[:final_k]

    def describe_active_pipeline(self) -> str:
        stages = ["Vector search"]
        if self.config.config.get("use_hybrid_search", True) and Document is not None:
            stages.append("FTS5 hybrid")
        if self.config.config.get("use_multi_query", False):
            stages.append(f"multi-query ×{self.config.config.get('multi_query_count', 3)}")
        if self.config.config.get("use_reranker", True) and CrossEncoder is not None:
            stages.append("cross-encoder rerank")
        return " → ".join(stages)
