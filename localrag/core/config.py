"""Settings: defaults, persistence to ~/.local_rag_app/config.json, local model resolution.
"""

import json
import os
from pathlib import Path

from localrag.core.network import is_offline_mode


def resolve_local_model(name_or_path: str, models_dir: Path, subdir: str) -> str:
    """Map an embedding / reranker model setting to a folder on disk.
    Accepts (1) a direct folder path, or (2) a HuggingFace-style name
    ('org/name') that has been copied into <models_dir>/<subdir>/ as 'name',
    'org/name', 'org--name' or 'org__name'. Falls back to the name unchanged
    (which then only works if it is already in the local HuggingFace cache)."""
    raw = (name_or_path or "").strip()
    if not raw:
        return raw
    direct = Path(raw).expanduser()
    if direct.is_dir():
        return str(direct)
    base = models_dir / subdir
    for candidate in (
        base / raw,
        base / raw.split("/")[-1],
        base / raw.replace("/", "--"),
        base / raw.replace("/", "__"),
    ):
        if candidate.is_dir():
            return str(candidate)
    return raw


def _default_config_values(models_dir: Path) -> dict:
    """Single source of truth for default settings (used for first run and
    for filling in keys missing from an older config.json)."""
    return {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model_path": str(models_dir / "llama-2-7b-chat.Q4_K_M.gguf"),
        "llm_context_length": 4096,
        "llm_max_tokens": 512,
        "llm_temperature": 0.1,
        "llm_gpu_layers": 0,
        "batch_size": 32,
        "max_docs_before_batch": 100,
        "use_hybrid_search": True,
        "hybrid_vector_weight": 0.6,
        "use_reranker": True,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "rerank_candidates": 20,
        "final_k": 5,
        "use_multi_query": False,
        "multi_query_count": 3,
        "language": "en",
        "last_bulk_folder_path": "",
        "use_chat_memory": True,
        "memory_turns": 3,
        "use_answer_cache": True,
        "chunking_strategy": "fixed",
        "ocr_enabled": True,
        "ocr_languages": "eng",
        "theme": "dark",
        "llm_backend": "gguf_only",
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "openai_base_url": "",
        "openai_provider": "openai",
        "openai_extra_headers": "",
        "openai_use_streaming": True,
        "network_mode": "offline" if is_offline_mode() else "online",
        # Which documents answers may use: "all" or "selected"
        "doc_scope_mode": "all",
        "doc_scope_files": [],
    }


class Config:
    """Configuration management for the RAG application"""

    def __init__(self):
        self.base_dir = Path.home() / ".local_rag_app"
        self.base_dir.mkdir(exist_ok=True, parents=True)

        self.docs_dir = self.base_dir / "documents"
        self.docs_dir.mkdir(exist_ok=True)

        self.db_dir = self.base_dir / "chroma_db"
        self.db_dir.mkdir(exist_ok=True)

        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        # Offline: models are never downloaded by the app — drop them here.
        (self.models_dir / "embeddings").mkdir(exist_ok=True)
        (self.models_dir / "rerankers").mkdir(exist_ok=True)

        self.config_file = self.base_dir / "config.json"
        self.load_config()

    def load_config(self):
        defaults = _default_config_values(self.models_dir)
        if not self.config_file.exists():
            self.config = defaults
            self.save_config()
        else:
            with open(self.config_file, "r") as f:
                self.config = json.load(f)
            changed = False
            for key, value in defaults.items():
                if key not in self.config:
                    self.config[key] = value
                    changed = True
            if changed:
                self.save_config()

    def save_config(self):
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        # The file can now hold an API key, so restrict it to the owner
        # (no-op / ignored on Windows).
        try:
            os.chmod(self.config_file, 0o600)
        except OSError:
            pass
