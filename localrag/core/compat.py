"""Resilient optional imports.

Every third-party RAG dependency is imported through _try_import so a missing
package becomes an entry in IMPORT_ERRORS (shown as a banner in the UI)
instead of crashing the app. Other modules import the resolved names from here.
"""


# Resilient imports
# --------------------------------------------------------------------------
IMPORT_ERRORS = []


def _try_import(label: str, install_hint: str, fn):
    try:
        return fn()
    except Exception as e:
        IMPORT_ERRORS.append((label, install_hint, f"{type(e).__name__}: {e}"))
        return None


_LOADER_IMPORT_SPECS = [
    ("PyPDFLoader", ".pdf", "langchain_community.document_loaders", "PyPDFLoader", "pypdf"),
    ("Docx2txtLoader", ".docx/.doc", "langchain_community.document_loaders", "Docx2txtLoader", "docx2txt"),
    ("TextLoader", ".txt", "langchain_community.document_loaders", "TextLoader", None),
    ("CSVLoader", ".csv", "langchain_community.document_loaders", "CSVLoader", None),
    ("UnstructuredExcelLoader", ".xlsx/.xls", "langchain_community.document_loaders", "UnstructuredExcelLoader", "unstructured[xlsx]"),
    ("UnstructuredPowerPointLoader", ".pptx/.ppt", "langchain_community.document_loaders", "UnstructuredPowerPointLoader", "unstructured[pptx]"),
    ("UnstructuredMarkdownLoader", ".md", "langchain_community.document_loaders", "UnstructuredMarkdownLoader", "unstructured[md]"),
    ("UnstructuredHTMLLoader", ".html/.htm", "langchain_community.document_loaders", "UnstructuredHTMLLoader", "unstructured[html]"),
    ("UnstructuredEPubLoader", ".epub", "langchain_community.document_loaders", "UnstructuredEPubLoader", "unstructured[epub]"),
    ("UnstructuredODTLoader", ".odt", "langchain_community.document_loaders", "UnstructuredODTLoader", "unstructured[odt]"),
    ("UnstructuredRTFLoader", ".rtf", "langchain_community.document_loaders", "UnstructuredRTFLoader", "unstructured[rtf]"),
    ("UnstructuredEmailLoader", ".eml/.msg", "langchain_community.document_loaders", "UnstructuredEmailLoader", "unstructured[email]"),
    ("JSONLoader", ".json", "langchain_community.document_loaders", "JSONLoader", "jq"),
]

_LOADER_EXT_MAP = {
    "PyPDFLoader": [".pdf"],
    "Docx2txtLoader": [".docx", ".doc"],
    "TextLoader": [".txt"],
    "CSVLoader": [".csv"],
    "UnstructuredExcelLoader": [".xlsx", ".xls"],
    "UnstructuredPowerPointLoader": [".pptx", ".ppt"],
    "UnstructuredMarkdownLoader": [".md"],
    "UnstructuredHTMLLoader": [".html", ".htm"],
    "UnstructuredEPubLoader": [".epub"],
    "UnstructuredODTLoader": [".odt"],
    "UnstructuredRTFLoader": [".rtf"],
    "UnstructuredEmailLoader": [".eml", ".msg"],
    "JSONLoader": [".json"],
}

LOADED_CLASSES = {}
for _cls_name, _exts_label, _module, _attr, _hint in _LOADER_IMPORT_SPECS:
    def _do_import(module=_module, attr=_attr):
        mod = __import__(module, fromlist=[attr])
        return getattr(mod, attr)

    result = _try_import(
        f"{_cls_name} ({_exts_label} support)",
        f"pip install {_hint} --break-system-packages" if _hint else "pip install langchain-community --break-system-packages",
        _do_import,
    )
    if result is not None:
        LOADED_CLASSES[_cls_name] = result

ALL_KNOWN_EXTENSIONS = {ext for exts in _LOADER_EXT_MAP.values() for ext in exts}

SUPPORTED_EXTENSIONS = {}
for _cls_name, _exts in _LOADER_EXT_MAP.items():
    if _cls_name in LOADED_CLASSES:
        for _ext in _exts:
            SUPPORTED_EXTENSIONS[_ext] = LOADED_CLASSES[_cls_name]

RecursiveCharacterTextSplitter = _try_import(
    "RecursiveCharacterTextSplitter",
    "pip install langchain-text-splitters --break-system-packages",
    lambda: __import__("langchain_text_splitters", fromlist=["RecursiveCharacterTextSplitter"]).RecursiveCharacterTextSplitter,
)

HuggingFaceEmbeddings = _try_import(
    "HuggingFaceEmbeddings",
    "pip install langchain-huggingface --break-system-packages",
    lambda: __import__("langchain_huggingface", fromlist=["HuggingFaceEmbeddings"]).HuggingFaceEmbeddings,
)
Chroma = _try_import(
    "Chroma",
    "pip install langchain-chroma --break-system-packages",
    lambda: __import__("langchain_chroma", fromlist=["Chroma"]).Chroma,
)

LlamaCpp = _try_import(
    "LlamaCpp",
    "pip install llama-cpp-python --break-system-packages",
    lambda: __import__("langchain_community.llms", fromlist=["LlamaCpp"]).LlamaCpp,
)
_callbacks_mod = _try_import(
    "CallbackManager / StreamingStdOutCallbackHandler",
    "pip install langchain-core --break-system-packages",
    lambda: __import__("langchain_core.callbacks", fromlist=["CallbackManager", "StreamingStdOutCallbackHandler"]),
)
CallbackManager = getattr(_callbacks_mod, "CallbackManager", None) if _callbacks_mod else None
StreamingStdOutCallbackHandler = getattr(_callbacks_mod, "StreamingStdOutCallbackHandler", None) if _callbacks_mod else None

RetrievalQA = _try_import(
    "RetrievalQA (langchain_classic)",
    "pip install langchain-classic --break-system-packages",
    lambda: __import__("langchain_classic.chains", fromlist=["RetrievalQA"]).RetrievalQA,
)
if RetrievalQA is None:
    RetrievalQA = _try_import(
        "RetrievalQA (langchain.chains fallback)",
        "pip install langchain --break-system-packages",
        lambda: __import__("langchain.chains", fromlist=["RetrievalQA"]).RetrievalQA,
    )
    if RetrievalQA is not None:
        IMPORT_ERRORS[:] = [e for e in IMPORT_ERRORS if not e[0].startswith("RetrievalQA")]

PromptTemplate = _try_import(
    "PromptTemplate",
    "pip install langchain-core --break-system-packages",
    lambda: __import__("langchain_core.prompts", fromlist=["PromptTemplate"]).PromptTemplate,
)

Document = _try_import(
    "Document (schema)",
    "pip install langchain-core --break-system-packages",
    lambda: __import__("langchain_core.documents", fromlist=["Document"]).Document,
)

BM25Retriever = _try_import(
    "BM25Retriever (keyword hybrid search)",
    "pip install rank_bm25 --break-system-packages",
    lambda: __import__("langchain_community.retrievers", fromlist=["BM25Retriever"]).BM25Retriever,
)

EnsembleRetriever = _try_import(
    "EnsembleRetriever (langchain_classic)",
    "pip install langchain-classic --break-system-packages",
    lambda: __import__("langchain_classic.retrievers", fromlist=["EnsembleRetriever"]).EnsembleRetriever,
)
if EnsembleRetriever is None:
    EnsembleRetriever = _try_import(
        "EnsembleRetriever (langchain.retrievers fallback)",
        "pip install langchain --break-system-packages",
        lambda: __import__("langchain.retrievers", fromlist=["EnsembleRetriever"]).EnsembleRetriever,
    )
    if EnsembleRetriever is not None:
        IMPORT_ERRORS[:] = [e for e in IMPORT_ERRORS if not e[0].startswith("EnsembleRetriever (langchain_classic)")]

CrossEncoder = _try_import(
    "CrossEncoder (reranking)",
    "pip install sentence-transformers --break-system-packages",
    lambda: __import__("sentence_transformers", fromlist=["CrossEncoder"]).CrossEncoder,
)

# --------------------------------------------------------------------------
# OCR fallback for scanned / image-only PDFs. PyPDFLoader extracts embedded
# text only — a scanned PDF has no text layer at all, so it would otherwise
# silently fail to index ("Loader ran but returned no usable text"). PyMuPDF
# renders each page to an image without needing the poppler system binary
# that pdf2image would require; pytesseract still needs the tesseract-ocr
# system package installed, but both Python packages degrade gracefully
# (OCR is just skipped, same as any other optional loader) if unavailable.
# --------------------------------------------------------------------------
PyMuPDFModule = _try_import(
    "PyMuPDF (OCR page rendering)",
    "pip install pymupdf --break-system-packages",
    lambda: __import__("fitz"),
)
Pytesseract = _try_import(
    "pytesseract (OCR text recognition — also needs the tesseract-ocr system package)",
    "pip install pytesseract --break-system-packages (then install the tesseract-ocr system package)",
    lambda: __import__("pytesseract"),
)
OCR_AVAILABLE = PyMuPDFModule is not None and Pytesseract is not None

# OpenAI SDK, for the optional cloud LLM backend that sits behind the
# offline GGUF model. Also used for any OpenAI-compatible server via a
# custom base URL (LM Studio, vLLM, OpenRouter, ...).
OpenAIModule = _try_import(
    "openai (optional cloud LLM backend)",
    "pip install openai --break-system-packages",
    lambda: __import__("openai"),
)
