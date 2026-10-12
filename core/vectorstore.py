"""Chroma vector store + embedding model management (supports per-document filtering).
"""

import logging
import shutil
from typing import List, Optional

from localrag.core.compat import Chroma, HuggingFaceEmbeddings
from localrag.core.config import Config, resolve_local_model
from localrag.core.network import is_offline_mode

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector store operations"""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self._initialize_embeddings()

    def _initialize_embeddings(self, model_name: str = None):
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "langchain-huggingface is not installed. "
                "Run: pip install langchain-huggingface --break-system-packages"
            )
        name = model_name or self.config.config["embedding_model"]
        resolved = resolve_local_model(name, self.config.models_dir, "embeddings")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=resolved,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(f"Embeddings initialized successfully ({resolved})")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            if is_offline_mode():
                raise RuntimeError(
                    f"Embedding model '{name}' was not found locally, and offline mode does not "
                    f"allow downloading it. Copy the model folder into "
                    f"{self.config.models_dir / 'embeddings'} (e.g. as '{name.split('/')[-1]}'), "
                    f"set a local folder path in Settings → Embeddings & Chunking, or switch to "
                    f"Online mode in the sidebar to let it download. "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            raise

    def create_or_update_vectorstore(self, documents, progress_callback=None):
        if Chroma is None:
            logger.error("langchain-chroma is not installed.")
            return False
        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.config.db_dir),
                    collection_name="documents",
                )
                logger.info(f"Created new vector store with {len(documents)} documents")
            else:
                self.vectorstore.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to existing vector store")

            return True
        except Exception as e:
            logger.error(f"Error creating/updating vector store: {e}")
            return False

    def load_vectorstore(self):
        if Chroma is None:
            return False
        try:
            if self.config.db_dir.exists():
                self.vectorstore = Chroma(
                    persist_directory=str(self.config.db_dir),
                    embedding_function=self.embeddings,
                    collection_name="documents",
                )
                count = self.vectorstore._collection.count()
                logger.info(f"Loaded vector store with {count} documents")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
        return False

    def get_retriever(self, k=5, file_names: Optional[List[str]] = None):
        """Vector retriever. If file_names is given, only chunks whose
        'file_name' metadata is in that list are searched."""
        if self.vectorstore:
            search_kwargs = {"k": k}
            if file_names:
                search_kwargs["filter"] = {"file_name": {"$in": list(file_names)}}
            return self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs=search_kwargs
            )
        return None

    def delete_documents_by_filename(self, file_name: str) -> bool:
        if not self.vectorstore:
            return False
        try:
            self.vectorstore._collection.delete(where={"file_name": file_name})
            return True
        except Exception as e:
            logger.error(f"Error deleting chunks for {file_name}: {e}")
            return False

    def clear_vectorstore(self):
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
            if self.config.db_dir.exists():
                shutil.rmtree(self.config.db_dir)
                self.config.db_dir.mkdir(exist_ok=True)
            self.vectorstore = None
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def get_document_count(self):
        if self.vectorstore:
            try:
                return self.vectorstore._collection.count()
            except Exception:
                pass
        return 0
