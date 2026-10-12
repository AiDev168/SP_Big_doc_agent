"""Local Document Q&A (RAG) package.

Importing this package configures logging and, via core.network, installs the
offline socket guard BEFORE any other module (HuggingFace, openai, chromadb...)
can open a connection.
"""
import logging

logging.basicConfig(level=logging.INFO)

from localrag.core import network  # noqa: E402,F401  (must be first: installs the guard)
