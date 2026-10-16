"""RAGApplication: wires config, documents, vector store, retrieval and LLM together.
"""

import logging
import re
import time
from pathlib import Path
from typing import List, Optional

from localrag.core.config import Config
from localrag.core.documents import DocumentManager
from localrag.core.llm import LLMManager
from localrag.core.processing import DocumentProcessor
from localrag.core.retrieval import AdvancedRetriever
from localrag.core.store import PersistentStore
from localrag.core.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGApplication:
    """Main RAG Application class"""

    def __init__(self):
        self.config = Config()
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_manager = VectorStoreManager(self.config)
        self.llm_manager = LLMManager(self.config)
        self.doc_manager = DocumentManager(self.config)
        self.persistent_store = PersistentStore(self.config)
        self.retriever = AdvancedRetriever(self.config, self.vector_manager, self.persistent_store)

        self.vector_manager.load_vectorstore()

    def add_documents(self, uploaded_files, progress_callback=None):
        saved_paths = []

        for uploaded_file in uploaded_files:
            path = self.doc_manager.save_uploaded_file(uploaded_file)
            if path:
                saved_paths.append(path)

        if not saved_paths:
            return False, "No files were saved", []

        try:
            documents, failures = self.doc_processor.load_documents(saved_paths, progress_callback)

            if not documents:
                return False, "No documents could be loaded", failures

            if progress_callback:
                progress_callback(0, 1, "Splitting documents into chunks...")

            chunks = self.doc_processor.split_documents(documents)

            if progress_callback:
                progress_callback(0, 1, f"Creating embeddings for {len(chunks)} chunks...")

            success = self.vector_manager.create_or_update_vectorstore(chunks, progress_callback)

            if success:
                self.persistent_store.append(chunks)
                for p in saved_paths:
                    if p.name not in {f for f, _ in failures}:
                        self.persistent_store.mark_file(p, status="done", chunk_count=0)
                self.retriever.mark_dirty()

            loaded_count = len(saved_paths) - len(failures)
            if success:
                if failures:
                    return True, f"Added {loaded_count} of {len(saved_paths)} files ({len(chunks)} chunks). {len(failures)} file(s) were skipped.", failures
                return True, f"Successfully added {loaded_count} files ({len(chunks)} chunks)", failures
            else:
                return False, "Failed to create vector store (embedding model or Chroma DB error — check the import banner and logs)", failures

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False, str(e), []

    def initialize_llm(self, model_path: str = None):
        return self.llm_manager.initialize_llm(model_path)

    def _corpus_fingerprint(self) -> str:
        """Cheap signature of 'what would change the answer to this exact
        question': the chunk count (changes on add/delete/clear) plus every
        retrieval-affecting setting, plus which LLM backend/model answers.
        Used to invalidate cached answers without needing to hash the whole
        corpus."""
        cfg = self.config.config
        backend = cfg.get("llm_backend", "gguf_only")
        parts = [
            str(self.vector_manager.get_document_count()),
            cfg.get("embedding_model", ""),
            str(cfg.get("use_hybrid_search", True)),
            str(cfg.get("hybrid_vector_weight", 0.6)),
            str(cfg.get("use_reranker", True)),
            cfg.get("reranker_model", ""),
            str(cfg.get("rerank_candidates", 20)),
            str(cfg.get("final_k", 5)),
            str(cfg.get("use_multi_query", False)),
            str(cfg.get("multi_query_count", 3)),
            backend,
            cfg.get("openai_model", "") if backend != "gguf_only" else "",
            cfg.get("openai_base_url", "") if backend != "gguf_only" else "",
        ]
        return "|".join(parts)

    @staticmethod
    def _cache_key(question: str, language: str, file_names: Optional[List[str]] = None) -> str:
        """The document scope is part of the key, so an answer cached for
        'all documents' is never served for a different selection."""
        import hashlib
        normalized = " ".join(question.strip().lower().split())
        scope = "|".join(sorted(file_names)) if file_names else "*"
        return hashlib.sha256(f"{language}::{scope}::{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _build_history_text(chat_history: Optional[List[dict]], max_turns: int) -> str:
        """Render the last max_turns user/assistant exchanges as plain text
        for the prompt. Kept short deliberately — this is context for
        resolving references, not material to answer from, so it shouldn't
        compete with the retrieved document chunks for the model's
        attention or eat into a small context window."""
        if not chat_history or max_turns <= 0:
            return ""
        recent = chat_history[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role_label = "User" if msg.get("role") == "user" else "Assistant"
            content = (msg.get("content") or "").strip()
            if len(content) > 300:
                content = content[:300] + "…"
            if content:
                lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _build_sources(docs: List, question: str) -> List[dict]:
        """Build rich source entries (file, page, and a short excerpt with
        query terms bolded) instead of just a bare filename — this is what
        lets someone actually verify an answer against the source text
        without having to go dig up the original document."""
        query_terms = [t_ for t_ in re.findall(r"\w+", question, flags=re.UNICODE) if len(t_) > 2]
        seen = set()
        entries = []
        for doc in docs:
            file_name = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "")
            key = (file_name, page)
            if key in seen:
                continue
            seen.add(key)

            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220] + "…"
            highlighted = snippet
            for term in sorted(set(query_terms), key=len, reverse=True)[:12]:
                try:
                    highlighted = re.sub(
                        rf"\b({re.escape(term)})\b", r"**\1**", highlighted, flags=re.IGNORECASE | re.UNICODE
                    )
                except re.error:
                    continue

            entries.append({
                "file_name": file_name,
                "page": page,
                "snippet": highlighted,
                "ocr": bool(doc.metadata.get("ocr")),
            })
        return entries

    def query(self, question: str, language: str = "en", file_names: Optional[List[str]] = None):
        if not self.llm_manager.is_ready():
            return None, "LLM not initialized. Please load the LLM model first."

        try:
            docs = self.retriever.retrieve(question, llm=self.llm_manager, file_names=file_names)

            if not docs:
                no_result_msg = (
                    "چیزی مرتبط با این پرسش در اسناد شما پیدا نکردم."
                    if language == "fa"
                    else "I couldn't find anything relevant to that in your documents."
                )
                return {
                    "result": no_result_msg,
                    "source_documents": [],
                }, None

            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = self.llm_manager.build_prompt(context, question, language=language)
            answer = self.llm_manager.invoke(prompt)
            if not isinstance(answer, str):
                answer = str(answer)

            return {"result": answer.strip(), "source_documents": docs}, None
        except Exception as e:
            logger.error(f"Error querying: {e}")
            return None, str(e)

    def query_stream(self, question: str, language: str = "en",
                     chat_history: Optional[List[dict]] = None,
                     file_names: Optional[List[str]] = None):
        """Generator version of query(): yields dicts as the answer is
        produced so the UI can show tokens as they arrive and report where
        time actually went, instead of one silent spinner covering both
        retrieval and generation. Also answers instantly from a persistent
        cache when the exact question was asked before and the corpus/
        retrieval settings haven't changed since.

        file_names: restrict retrieval to these documents (None = all).

        Yields, in order:
          {"type": "meta", "sources": [...], "retrieval_time": float, "from_cache": bool}
          {"type": "token", "text": str}                (repeated)
          {"type": "done", "answer": str, "generation_time": float, "total_time": float,
           "backend": "gguf" | "openai" | "cache" | None}
        or, on failure:
          {"type": "error", "message": str}
        sources are dicts: {"file_name", "page", "snippet", "ocr"}.
        """
        if not self.llm_manager.is_ready():
            yield {"type": "error", "message": "LLM not initialized. Please load the LLM model first."}
            return

        t_start = time.time()
        use_cache = self.config.config.get("use_answer_cache", True)
        cache_key = self._cache_key(question, language, file_names)
        fingerprint = self._corpus_fingerprint()

        if use_cache:
            cached = self.persistent_store.get_cached_answer(cache_key, fingerprint)
            if cached is not None:
                yield {"type": "meta", "sources": cached["sources"], "retrieval_time": 0.0, "from_cache": True}
                yield {"type": "token", "text": cached["answer"]}
                yield {
                    "type": "done",
                    "answer": cached["answer"],
                    "generation_time": 0.0,
                    "total_time": time.time() - t_start,
                    "backend": "cache",
                }
                return

        try:
            docs = self.retriever.retrieve(question, llm=self.llm_manager, file_names=file_names)
        except Exception as e:
            logger.error(f"Error retrieving: {e}")
            yield {"type": "error", "message": str(e)}
            return
        retrieval_time = time.time() - t_start

        if not docs:
            no_result_msg = (
                "چیزی مرتبط با این پرسش در اسناد شما پیدا نکردم."
                if language == "fa"
                else "I couldn't find anything relevant to that in your documents."
            )
            yield {"type": "meta", "sources": [], "retrieval_time": retrieval_time, "from_cache": False}
            yield {"type": "token", "text": no_result_msg}
            yield {
                "type": "done", "answer": no_result_msg, "generation_time": 0.0,
                "total_time": time.time() - t_start, "backend": None,
            }
            return

        sources = self._build_sources(docs, question)

        context = "\n\n".join(doc.page_content for doc in docs)
        history_text = ""
        if self.config.config.get("use_chat_memory", True):
            history_text = self._build_history_text(chat_history, int(self.config.config.get("memory_turns", 3)))
        prompt = self.llm_manager.build_prompt(context, question, language=language, history_text=history_text)

        yield {"type": "meta", "sources": sources, "retrieval_time": retrieval_time, "from_cache": False}

        t_gen_start = time.time()
        chunks = []
        try:
            for text in self.llm_manager.stream(prompt):
                chunks.append(text)
                yield {"type": "token", "text": text}
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield {"type": "error", "message": str(e)}
            return
        generation_time = time.time() - t_gen_start
        answer = "".join(chunks).strip()

        if use_cache and answer:
            try:
                self.persistent_store.set_cached_answer(cache_key, question, answer, sources, fingerprint)
            except Exception:
                logger.exception("Failed to write answer cache")

        yield {
            "type": "done",
            "answer": answer,
            "generation_time": generation_time,
            "total_time": time.time() - t_start,
            "backend": self.llm_manager.last_backend_used,
        }

    def delete_document_full(self, file_path: Path) -> bool:
        file_name = file_path.name
        disk_ok = self.doc_manager.delete_document(file_path)
        index_ok = self.vector_manager.delete_documents_by_filename(file_name)
        self.persistent_store.remove_by_filename(file_name)
        self.retriever.mark_dirty()
        return disk_ok and (index_ok or self.vector_manager.get_document_count() == 0)

    def clear_knowledge_base(self):
        try:
            self.vector_manager.clear_vectorstore()

            for doc in self.doc_manager.get_all_documents():
                self.doc_manager.delete_document(doc)

            self.persistent_store.clear()
            self.retriever.mark_dirty()
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False

    def get_manifest_stats(self) -> dict:
        return self.persistent_store.manifest_stats()
