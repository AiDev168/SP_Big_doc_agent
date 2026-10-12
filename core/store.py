"""SQLite store: chunk corpus + FTS5 keyword index, ingestion manifest, answer cache.
"""

import json
import logging
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from localrag.core.compat import Document
from localrag.core.config import Config

logger = logging.getLogger(__name__)


def _normalize_search_text(text: str) -> str:
    """Normalize text before it goes into (or is queried against) the FTS5
    keyword index. Persian text in particular varies a lot in ways that
    would otherwise break exact keyword matching: the zero-width
    non-joiner used inside compound words, and Arabic-vs-Persian forms of
    the same letters (ي/ی, ك/ک) that different keyboards/fonts produce."""
    if not text:
        return text
    text = text.replace("\u200c", " ")  # ZWNJ -> space
    text = text.replace("\u064a", "\u06cc").replace("\u0643", "\u06a9")  # Arabic ye/kaf -> Persian
    return text


def _fts5_match_expression(query: str) -> Optional[str]:
    """Build a safe FTS5 MATCH expression from free-text user input.
    Tokens are individually double-quoted so punctuation, parentheses, or
    other FTS5 query-syntax characters in the user's question can't raise a
    syntax error — each token is matched as a literal phrase, OR'd together."""
    query = _normalize_search_text(query)
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    tokens = [tok for tok in tokens if tok]
    if not tokens:
        return None
    escaped = [tok.replace('"', '""') for tok in tokens[:32]]
    return " OR ".join(f'"{tok}"' for tok in escaped)


class PersistentStore:
    """Disk-backed, incrementally-updatable storage for two things that used
    to live only in memory or in an ever-growing flat file:

      1. The raw chunk corpus + a full-text (FTS5) keyword index, used for
         hybrid search. SQLite FTS5 is C-implemented and disk-backed, so
         keyword search scales to a large corpus without ever loading every
         chunk into Python memory — unlike the previous JSONL + in-memory
         rank_bm25 approach, which had to read and re-tokenize the entire
         corpus on every single change.
      2. An ingestion manifest recording which source files have already
         been indexed (path, size, mtime, status, chunk count). This is
         what makes bulk folder ingestion resumable: if a 50GB import is
         interrupted, restarting it skips every file already marked done
         instead of re-processing the whole corpus.

    One SQLite connection (WAL mode, so reads from the UI don't block an
    ingestion write in progress) backs both concerns.
    """

    def __init__(self, config: Config):
        self.config = config
        self.path = config.db_dir / "store.sqlite3"
        self._local = threading.local()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        # A connection per thread: the background ingestion thread and the
        # Streamlit UI thread must not share a single sqlite3 connection.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                source_file TEXT,
                metadata_json TEXT,
                page_content TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_name ON chunks(file_name);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                page_content,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, page_content) VALUES (new.id, new.page_content);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, page_content) VALUES('delete', old.id, old.page_content);
            END;

            CREATE TABLE IF NOT EXISTS indexed_files (
                path TEXT PRIMARY KEY,
                file_name TEXT,
                size INTEGER,
                mtime REAL,
                status TEXT,
                chunk_count INTEGER DEFAULT 0,
                error TEXT,
                indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS answer_cache (
                cache_key TEXT PRIMARY KEY,
                question TEXT,
                answer TEXT,
                sources_json TEXT,
                corpus_fingerprint TEXT,
                created_at TEXT
            );
            """
        )
        conn.commit()

    # -- answer cache --------------------------------------------------------

    def get_cached_answer(self, cache_key: str, corpus_fingerprint: str) -> Optional[dict]:
        """Return the cached {answer, sources} for this exact question if
        present *and* the corpus hasn't changed since it was cached — a
        cheap chunk-count-based fingerprint is enough to avoid ever serving
        a stale answer after documents are added, removed, or the knowledge
        base is cleared, without needing anything more elaborate."""
        conn = self._connect()
        row = conn.execute(
            "SELECT answer, sources_json, corpus_fingerprint FROM answer_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        answer, sources_json, cached_fingerprint = row
        if cached_fingerprint != corpus_fingerprint:
            return None
        try:
            sources = json.loads(sources_json) if sources_json else []
        except Exception:
            sources = []
        return {"answer": answer, "sources": sources}

    def set_cached_answer(self, cache_key: str, question: str, answer: str, sources: list, corpus_fingerprint: str) -> None:
        conn = self._connect()
        conn.execute(
            """
            INSERT INTO answer_cache (cache_key, question, answer, sources_json, corpus_fingerprint, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                answer=excluded.answer, sources_json=excluded.sources_json,
                corpus_fingerprint=excluded.corpus_fingerprint, created_at=excluded.created_at
            """,
            (cache_key, question, answer, json.dumps(sources, ensure_ascii=False), corpus_fingerprint, datetime.now().isoformat()),
        )
        conn.commit()

    def clear_answer_cache(self) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM answer_cache")
        conn.commit()

    def answer_cache_count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0]

    # -- chunk storage / keyword search -----------------------------------

    def append(self, documents) -> None:
        conn = self._connect()
        rows = [
            (
                doc.metadata.get("file_name", ""),
                doc.metadata.get("source_file", ""),
                json.dumps(doc.metadata, ensure_ascii=False),
                _normalize_search_text(doc.page_content),
            )
            for doc in documents
        ]
        conn.executemany(
            "INSERT INTO chunks (file_name, source_file, metadata_json, page_content) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()

    def remove_by_filename(self, file_name: str) -> None:
        conn = self._connect()
        conn.execute("DELETE FROM chunks WHERE file_name = ?", (file_name,))
        conn.commit()

    def clear(self) -> None:
        conn = self._connect()
        conn.executescript(
            "DELETE FROM chunks; DELETE FROM chunks_fts; DELETE FROM indexed_files;"
        )
        conn.commit()

    def count(self) -> int:
        conn = self._connect()
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def list_file_names(self) -> List[str]:
        """Distinct file names that currently have indexed chunks (covers both
        uploaded files and bulk-imported folder files)."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT file_name FROM chunks WHERE file_name != '' ORDER BY file_name COLLATE NOCASE"
        ).fetchall()
        return [r[0] for r in rows]

    def search_keyword(self, query: str, k: int = 10, file_names: Optional[List[str]] = None) -> List:
        """FTS5 keyword search, ranked by SQLite's built-in bm25() function.
        Runs entirely on disk/index — never loads the corpus into memory.
        If file_names is given, only chunks from those files are searched."""
        if Document is None:
            return []
        match_expr = _fts5_match_expression(query)
        if not match_expr:
            return []
        conn = self._connect()
        sql = """
            SELECT c.page_content, c.metadata_json, bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params: list = [match_expr]
        if file_names:
            sql += " AND c.file_name IN (%s)" % ",".join("?" * len(file_names))
            params.extend(file_names)
        sql += " ORDER BY score LIMIT ?"
        params.append(k)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            logger.exception("FTS5 keyword search failed for query")
            return []
        results = []
        for page_content, metadata_json, _score in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {}
            results.append(Document(page_content=page_content, metadata=metadata))
        return results

    # -- ingestion manifest (resumable bulk import) ------------------------

    def is_up_to_date(self, path: Path) -> bool:
        """True if this exact file (by path, size and mtime) was already
        successfully indexed — lets a bulk import skip unchanged files on
        resume instead of re-processing a 50GB corpus from scratch."""
        conn = self._connect()
        row = conn.execute(
            "SELECT size, mtime, status FROM indexed_files WHERE path = ?",
            (str(path),),
        ).fetchone()
        if row is None:
            return False
        size, mtime, status = row
        try:
            st_ = path.stat()
        except OSError:
            return False
        return status == "done" and size == st_.st_size and abs(mtime - st_.st_mtime) < 1.0

    def mark_file(self, path: Path, status: str, chunk_count: int = 0, error: str = "") -> None:
        conn = self._connect()
        try:
            size = path.stat().st_size
            mtime = path.stat().st_mtime
        except OSError:
            size, mtime = 0, 0.0
        conn.execute(
            """
            INSERT INTO indexed_files (path, file_name, size, mtime, status, chunk_count, error, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                file_name=excluded.file_name, size=excluded.size, mtime=excluded.mtime,
                status=excluded.status, chunk_count=excluded.chunk_count,
                error=excluded.error, indexed_at=excluded.indexed_at
            """,
            (str(path), path.name, size, mtime, status, chunk_count, error, datetime.now().isoformat()),
        )
        conn.commit()

    def manifest_stats(self) -> dict:
        conn = self._connect()
        row = conn.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status='done') AS done,
                COUNT(*) FILTER (WHERE status='failed') AS failed,
                COALESCE(SUM(chunk_count) FILTER (WHERE status='done'), 0) AS chunks,
                COALESCE(SUM(size) FILTER (WHERE status='done'), 0) AS bytes
            FROM indexed_files
            """
        ).fetchone()
        return {"done": row[0] or 0, "failed": row[1] or 0, "chunks": row[2] or 0, "bytes": row[3] or 0}

    def failed_files(self, limit: int = 50) -> List[dict]:
        conn = self._connect()
        rows = conn.execute(
            "SELECT path, error, indexed_at FROM indexed_files WHERE status='failed' ORDER BY indexed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"path": r[0], "error": r[1], "indexed_at": r[2]} for r in rows]

    # Retained for compatibility with anything that still wants the full
    # corpus as Document objects (small corpora only — this does load
    # everything into memory, so bulk-imported 50GB corpora should use
    # search_keyword() instead, never this).
    def load_all(self) -> List:
        if Document is None:
            return []
        conn = self._connect()
        rows = conn.execute("SELECT page_content, metadata_json FROM chunks").fetchall()
        docs = []
        for page_content, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except Exception:
                metadata = {}
            docs.append(Document(page_content=page_content, metadata=metadata))
        return docs


# Backward-compatible alias — earlier revisions of this file called this
# class ChunkStore; keep the name usable in case anything external imports it.
ChunkStore = PersistentStore
