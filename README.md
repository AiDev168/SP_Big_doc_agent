# Local Document Q&A (RAG)

    pip install -r requirements.txt
    streamlit run app.py

## Layout

    app.py                     entry point (3 lines)
    localrag/
      __init__.py              logging + installs the offline guard FIRST
      core/                    no UI code — reusable / testable on its own
        network.py             Offline/Online mode, socket guard
        compat.py              resilient optional imports (IMPORT_ERRORS)
        constants.py           backend modes, provider presets, file icons
        config.py              defaults, config.json, resolve_local_model
        store.py               SQLite: chunks + FTS5, manifest, answer cache
        processing.py          loaders, OCR fallback, chunk splitting
        documents.py           uploaded-files folder management
        vectorstore.py         Chroma + embeddings (per-file filter)
        retrieval.py           multi-query, hybrid fusion, reranking
        llm.py                 GGUF + OpenAI-compatible backends, prompts
        application.py         RAGApplication (query / query_stream / ...)
        indexer.py             background bulk-folder indexing
      ui/
        i18n.py                translations, RTL helpers
        theme.py               CSS (dark/light, LTR/RTL)
        components.py          hero, empty state, chat bubble, sources, scope selector
        main.py                sidebar + page routing
        pages/ chat.py documents.py settings.py status.py

## Dependency direction

    network, compat, constants -> config -> store, processing, vectorstore
    -> retrieval, llm, documents -> application -> indexer
    ui/* depends on core/*; core/* never imports streamlit or ui/*.

## Where to change things

- New file type: `core/compat.py` (_LOADER_IMPORT_SPECS + _LOADER_EXT_MAP), icon in `core/constants.py`
- New UI language / label: `ui/i18n.py`
- New LLM provider preset: `core/constants.py`
- New retrieval stage: `core/retrieval.py`
- New page: add `ui/pages/<name>.py` and route it in `ui/main.py`
