"""Background bulk-folder ingestion (resumable, one file at a time).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List

from localrag.core.compat import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from localrag.core.application import RAGApplication

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Bulk folder ingestion — the path meant for large (multi-GB / tens-of-GB)
# corpora. Browser upload through st.file_uploader isn't realistic at that
# scale, so this reads files directly from a folder path on the machine
# running the app, in a background thread, one file at a time:
#   - per-file processing bounds peak memory regardless of corpus size
#   - each file is committed (embeddings + keyword index + manifest row)
#     before moving to the next, so a crash or a deliberate stop only ever
#     loses at most the one file in flight — everything else already done
#     is safely on disk and does not get reprocessed on resume
#   - progress is written to a JSON status file so the Streamlit UI (which
#     re-runs the whole script on every interaction) can poll it without
#     blocking on the ingestion itself
# --------------------------------------------------------------------------

# thread registry + stop flags, keyed by folder path — module-level (not
# st.session_state) because it must survive across Streamlit reruns and be
# reachable from a different thread than the one that started it.
_FOLDER_INDEX_REGISTRY: dict = {}
_FOLDER_INDEX_LOCK = threading.Lock()


class FolderIndexer:
    """Walks a folder, indexes every supported file not already up to date
    in the manifest, one file at a time, writing progress to a status file."""

    def __init__(self, app: "RAGApplication", folder_path: Path):
        self.app = app
        self.folder_path = folder_path
        self.status_path = app.config.db_dir / f"folder_import_status_{self._status_key(folder_path)}.json"

    @staticmethod
    def _status_key(folder_path: Path) -> str:
        # hashlib, not the built-in hash(): Python randomizes str hash()
        # per process for security, which would otherwise pick a different
        # status file name every restart and make progress look reset even
        # though the manifest (which drives actual resume behavior) is fine.
        import hashlib
        return hashlib.md5(str(folder_path.resolve()).encode("utf-8")).hexdigest()

    def _scan_folder(self) -> List[Path]:
        exts = tuple(SUPPORTED_EXTENSIONS.keys())
        files = []
        for root, _dirs, filenames in os.walk(self.folder_path):
            for fname in filenames:
                if fname.lower().endswith(exts):
                    files.append(Path(root) / fname)
        return sorted(files)

    def _write_status(self, **fields) -> None:
        current = {}
        if self.status_path.exists():
            try:
                current = json.loads(self.status_path.read_text(encoding="utf-8"))
            except Exception:
                current = {}
        current.update(fields)
        current["updated_at"] = datetime.now().isoformat()
        tmp_path = self.status_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(self.status_path)

    def read_status(self) -> dict:
        if not self.status_path.exists():
            return {"status": "idle"}
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "idle"}

    def run(self, stop_event: threading.Event) -> None:
        store = self.app.persistent_store
        try:
            files = self._scan_folder()
        except Exception as e:
            self._write_status(status="error", error=f"Could not scan folder: {e}")
            return

        pending = [f for f in files if not store.is_up_to_date(f)]
        self._write_status(
            status="running",
            total_files=len(files),
            pending_files=len(pending),
            done_files=len(files) - len(pending),
            current_file="",
            chunks_indexed=store.manifest_stats().get("chunks", 0),
            errors=0,
            started_at=datetime.now().isoformat(),
        )

        done_count = len(files) - len(pending)
        error_count = 0

        for path in pending:
            if stop_event.is_set():
                self._write_status(status="stopped", current_file="")
                return

            self._write_status(current_file=str(path))
            try:
                documents, failures = self.app.doc_processor.load_documents([path])
                if failures or not documents:
                    reason = failures[0][1] if failures else "No usable text extracted"
                    store.mark_file(path, status="failed", chunk_count=0, error=reason)
                    error_count += 1
                    continue

                chunks = self.app.doc_processor.split_documents(documents)
                ok = self.app.vector_manager.create_or_update_vectorstore(chunks)
                if ok:
                    store.append(chunks)
                    store.mark_file(path, status="done", chunk_count=len(chunks))
                else:
                    store.mark_file(path, status="failed", chunk_count=0, error="Failed to add to vector store")
                    error_count += 1

                del documents, chunks
                gc.collect()
            except Exception as e:
                logger.exception(f"Folder indexing failed on {path}")
                store.mark_file(path, status="failed", chunk_count=0, error=f"{type(e).__name__}: {e}")
                error_count += 1
            finally:
                done_count += 1
                self._write_status(
                    done_files=done_count,
                    errors=error_count,
                    chunks_indexed=store.manifest_stats().get("chunks", 0),
                )

        self._write_status(status="done", current_file="")


def start_folder_indexing(app: "RAGApplication", folder_path: Path) -> FolderIndexer:
    """Start (or return the already-running) background indexer for this
    folder. Safe to call repeatedly — e.g. from a Streamlit rerun — without
    spawning duplicate threads."""
    indexer = FolderIndexer(app, folder_path)
    key = str(folder_path.resolve())
    with _FOLDER_INDEX_LOCK:
        entry = _FOLDER_INDEX_REGISTRY.get(key)
        if entry and entry["thread"].is_alive():
            return indexer
        stop_event = threading.Event()
        thread = threading.Thread(target=indexer.run, args=(stop_event,), daemon=True)
        _FOLDER_INDEX_REGISTRY[key] = {"thread": thread, "stop_event": stop_event}
        thread.start()
    return indexer


def stop_folder_indexing(folder_path: Path) -> None:
    key = str(folder_path.resolve())
    with _FOLDER_INDEX_LOCK:
        entry = _FOLDER_INDEX_REGISTRY.get(key)
        if entry:
            entry["stop_event"].set()


def is_folder_indexing_running(folder_path: Path) -> bool:
    key = str(folder_path.resolve())
    with _FOLDER_INDEX_LOCK:
        entry = _FOLDER_INDEX_REGISTRY.get(key)
        return bool(entry and entry["thread"].is_alive())
