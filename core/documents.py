"""Management of the uploaded-documents folder on disk.
"""

import logging
from pathlib import Path
from typing import List, Optional

from localrag.core.config import Config
from localrag.core.processing import DocumentProcessor

logger = logging.getLogger(__name__)


class DocumentManager:
    """Manages documents in the application directory"""

    def __init__(self, config: Config):
        self.config = config
        self.docs_dir = config.docs_dir

    def save_uploaded_file(self, uploaded_file) -> Optional[Path]:
        try:
            file_path = self.docs_dir / uploaded_file.name

            counter = 1
            while file_path.exists():
                stem = Path(uploaded_file.name).stem
                suffix = Path(uploaded_file.name).suffix
                file_path = self.docs_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            logger.error(f"Error saving file {uploaded_file.name}: {e}")
            return None

    def get_all_documents(self) -> List[Path]:
        documents = []
        for ext in DocumentProcessor.SUPPORTED_EXTENSIONS.keys():
            documents.extend(self.docs_dir.glob(f"*{ext}"))
        return sorted(documents)

    def delete_document(self, file_path: Path) -> bool:
        try:
            if file_path.exists():
                file_path.unlink()
                return True
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
        return False

    def get_total_size(self) -> int:
        total = 0
        for doc in self.get_all_documents():
            total += doc.stat().st_size
        return total

    def format_size(self, size_bytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
